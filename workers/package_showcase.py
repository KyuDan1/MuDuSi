"""Join already-rendered clips without re-timing their speech or captions."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    import imageio_ffmpeg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('clips', nargs='+')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    clips = [Path(p).resolve() for p in args.clips]
    metadata = [json.loads(Path(str(p) + '.json').read_text()) for p in clips]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as listing:
        for clip in clips:
            if '\n' in str(clip):
                raise ValueError('Newline in clip path')
            listing.write("file '" + str(clip).replace("'", "'\\''") + "'\n")
        listing.flush()
        subprocess.run([ffmpeg, '-y', '-hide_banner', '-f', 'concat', '-safe', '0', '-i', listing.name,
                        '-c', 'copy', '-movflags', '+faststart', '-metadata', 'title=MuDuSi — Two and Three Agents',
                        '-metadata', 'comment=Real model emissions. Word-timed offline ASR. Recorded spatial interaction.', str(output)], check=True)
    checks = {}
    for clip in [*clips, output]:
        result = subprocess.run([ffmpeg, '-v', 'error', '-i', str(clip), '-map', '0:v:0', '-map', '0:a:0',
                                 '-threads', '4', '-progress', 'pipe:1', '-f', 'null', '-'], capture_output=True, text=True)
        if result.returncode or result.stderr.strip():
            raise RuntimeError(f'Video decode verification failed: {clip}: {result.stderr}')
        progress = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        checks[clip.name] = {'decode_ok': True, 'video_frames': int(progress['frame']),
                            'duration_seconds': int(progress['out_time_us']) / 1e6}
    Path(str(output) + '.json').write_text(json.dumps({'clips': metadata, 'verification': checks}, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
