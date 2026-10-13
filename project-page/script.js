// Keep the listening experience clear when moving between conversations.
const videos = [...document.querySelectorAll('video')];
for (const video of videos) {
  video.addEventListener('play', () => {
    for (const other of videos) if (other !== video) other.pause();
  });
}

const sectionLinks = [...document.querySelectorAll('.section-nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of sectionLinks) {
        if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
    }
  }, { rootMargin: '-5% 0px -65% 0px', threshold: 0 });
  document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
}

// Start silently; native controls enable sound and the button handles blocked autoplay.
const introVideo = document.querySelector('#introduction-video');
const introPlay = document.querySelector('.hero-film-play');
if (introVideo && introPlay) {
  introPlay.hidden = !introVideo.paused;
  introPlay.addEventListener('click', async () => {
    if (introVideo.ended) introVideo.currentTime = 0;
    introVideo.muted = false;
    try { await introVideo.play(); } catch { introPlay.hidden = false; }
  });
  introVideo.addEventListener('play', () => { introPlay.hidden = true; });
  introVideo.addEventListener('ended', () => { introPlay.hidden = false; });
  introVideo.muted = true;
  introVideo.play().catch(() => { introPlay.hidden = false; });
}
