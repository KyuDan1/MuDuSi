"""Compose chapter cards and recorded clips with speech-preserving crossfades."""
import argparse
import json
from pathlib import Path
import subprocess


def edit_plan(segments, transition=.5):
    if transition <= 0:
        raise ValueError('transition must be positive')
    plan = []
    end = 0.0
    for index, segment in enumerate(segments):
        kind, duration = segment['kind'], float(segment['seconds'])
        if kind not in ('card', 'video') or duration <= 0:
            raise ValueError('Expected a positive card/video duration')
        padding = transition if kind == 'video' else 0
        length = duration + 2 * padding
        if length < 2 * transition:
            raise ValueError('Card is too short for two transitions')
        start = end - transition if index else 0
        end = start + length
        plan.append({**segment, 'start': start, 'end': end, 'padded_seconds': length,
                     'speech_start': start + padding if kind == 'video' else None,
                     'speech_end': start + padding + duration if kind == 'video' else None})
    if not plan:
        raise ValueError('No segments')
    return plan


def video_frames(ffmpeg, path):
    result = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:v:0',
                             '-threads', '4', '-progress', 'pipe:1', '-f', 'null', '-'],
                            capture_output=True, text=True, check=True)
    progress = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    return int(progress['frame'])


def main():
    import imageio_ffmpeg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.manifest).read_text())
    fps, fade = int(spec.get('fps', 24)), float(spec.get('transition', .5))
    width, height = int(spec.get('width', 1280)), int(spec.get('height', 720))
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise ValueError('Video dimensions must be positive even integers')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = spec['segments']
    for segment in segments:
        if segment['kind'] == 'video':
            segment['seconds'] = video_frames(ffmpeg, segment['path']) / fps
    plan = edit_plan(segments, fade)
    command = [ffmpeg, '-y', '-hide_banner', '-filter_complex_threads', '4']
    filters = []
    for i, segment in enumerate(plan):
        kind, duration = segment['kind'], segment['seconds']
        if kind == 'card':
            command += ['-loop', '1', '-framerate', str(fps), '-t', str(duration)]
        command += ['-i', segment['path']]
        # setpts clears frame-rate metadata; restore fps before tpad/xfade.
        vf = f'[{i}:v]fps={fps},scale={width}:{height},setsar=1,format=yuv420p,trim=duration={duration},settb=1/{fps},setpts=N,fps={fps}'
        if kind == 'video':
            vf += f',tpad=start_mode=clone:stop_mode=clone:start_duration={fade}:stop_duration={fade}'
            filters.append(f'[{i}:a]atrim=duration={duration},asetpts=PTS-STARTPTS,aresample=48000,'
                           f'aformat=sample_fmts=fltp:channel_layouts=mono,adelay={int(fade*1000)}:all=1,'
                           f'apad=pad_dur={fade},atrim=duration={segment["padded_seconds"]}[a{i}]')
        else:
            filters.append(f'anullsrc=r=48000:cl=mono,atrim=duration={duration},asetpts=PTS-STARTPTS[a{i}]')
        filters.append(vf + f',settb=1/{fps},setpts=N,fps={fps}[v{i}]')
    previous_v, previous_a = 'v0', 'a0'
    for i in range(1, len(plan)):
        filters += [f'[{previous_v}][v{i}]xfade=transition=fade:duration={fade}:offset={plan[i]["start"]},fps={fps}[xv{i}]',
                    f'[{previous_a}][a{i}]acrossfade=d={fade}:c1=tri:c2=tri[xa{i}]']
        previous_v, previous_a = f'xv{i}', f'xa{i}'
    duration = plan[-1]['end']
    filters.append(f'[{previous_v}]fade=t=in:d={fade},fade=t=out:st={duration-fade}:d={fade}[video]')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command += ['-filter_complex', ';'.join(filters), '-map', '[video]', '-map', f'[{previous_a}]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '4', '-pix_fmt', 'yuv420p',
                '-r', str(fps), '-c:a', 'aac', '-ar', '48000', '-b:a', '192k', '-t', str(duration),
                '-movflags', '+faststart', '-metadata', 'title=' + spec.get('title', 'MuDuSi — Introduction'), str(output)]
    subprocess.run(command, check=True)
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(output), '-map', '0:v:0', '-map', '0:a:0',
                              '-threads', '4', '-progress', 'pipe:1', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    if decoded.stderr.strip():
        raise RuntimeError(decoded.stderr)
    progress = dict(line.split('=', 1) for line in decoded.stdout.splitlines() if '=' in line)
    if abs(int(progress['frame']) - round(duration * fps)) > 1:
        raise RuntimeError(f'Incomplete video: {progress["frame"]} frames, expected {round(duration * fps)}')
    result = {'width': width, 'height': height, 'fps': fps, 'transition': fade, 'planned_duration': duration, 'segments': plan,
              'decode_ok': True, 'decoded_frames': int(progress['frame']),
              'decoded_seconds': int(progress['out_time_us']) / 1e6,
              'audio': 'Original model audio; silent cards and padding protect speech during crossfades.'}
    Path(str(output)+'.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
