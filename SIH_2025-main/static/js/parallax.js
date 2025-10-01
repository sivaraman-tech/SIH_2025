document.addEventListener('scroll', function() {
  const scrolled = window.pageYOffset;
  const hero = document.querySelector('header');
  if (hero) {
    hero.style.backgroundPositionY = -(scrolled * 0.5) + 'px';
  }
});
