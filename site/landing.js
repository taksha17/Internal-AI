document.documentElement.classList.add('js');

// FAQ accordion
document.querySelectorAll('.faq-question').forEach(btn => {
  btn.addEventListener('click', () => {
    const expanded = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', !expanded);
    const answer = document.getElementById(btn.getAttribute('aria-controls'));
    answer.hidden = expanded;
  });
});

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const format = (value, decimals) =>
  value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

const revealables = [...document.querySelectorAll('.reveal')];
const tagline = document.querySelector('.tagline-reveal');
const words = [...document.querySelectorAll('.tagline-word')];
const figures = [...document.querySelectorAll('[data-count]')];

/* Relative to the viewport: true once the element's top has passed 92% of the
   window height. Geometry rather than IntersectionObserver, so a page rendered
   in a covered or unpainted window still resolves instead of staying blank. */
const reached = (el, cut) => {
  const box = el.getBoundingClientRect();
  const height = window.innerHeight || document.documentElement.clientHeight;
  return box.top < height * cut && box.bottom > 0;
};

function countUp(el) {
  const target = Number(el.dataset.count);
  const decimals = Number(el.dataset.decimals || 0);
  const start = performance.now();
  const duration = 1100;

  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = format(target * eased, decimals);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = format(target, decimals);
  };

  requestAnimationFrame(step);
}

function paint() {
  pending = false;

  revealables.forEach(el => {
    if (!el.classList.contains('visible') && reached(el, 0.94)) {
      el.classList.add('visible');
    }
  });

  if (tagline && words.length && reached(tagline, 0.9)) {
    words.forEach(word => word.classList.add('visible'));
  }

  figures.forEach(el => {
    if (el.dataset.counted || !reached(el, 0.92)) return;
    el.dataset.counted = '1';
    countUp(el);
  });
}

let pending = false;
const schedule = () => {
  if (pending) return;
  pending = true;
  requestAnimationFrame(paint);
};

if (reduced) {
  revealables.forEach(el => el.classList.add('visible'));
  words.forEach(word => word.classList.add('visible'));
} else {
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  paint();
}
