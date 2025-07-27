(function () {
  const templates = {
    router: document.getElementById('router-template'),
    energy: document.getElementById('energy-template'),
    compressor: document.getElementById('compressor-template'),
    decompressor: document.getElementById('decompressor-template'),
    hpo: document.getElementById('hpo-template'),
    nvlink: document.getElementById('nvlink-template'),
    qattention: document.getElementById('qattention-template'),
  };

  const workspace = document.getElementById('workspace');

  function loadModule(name) {
    workspace.textContent = '';
    const tpl = templates[name];
    if (!tpl) return;
    const clone = tpl.content.cloneNode(true);
    workspace.appendChild(clone);
    const mod = window.modules && window.modules[name];
    if (mod && typeof mod.init === 'function') mod.init(workspace);
    const section = workspace.querySelector('.module');
    if (section) {
      requestAnimationFrame(() => section.classList.add('visible'));
    }
  }

  document.getElementById('nav').addEventListener('click', (e) => {
    const item = e.target.closest('li[data-module]');
    if (!item) return;
    document.querySelectorAll('#nav li').forEach(li => li.classList.remove('active'));
    item.classList.add('active');
    loadModule(item.dataset.module);
  });

  setInterval(() => {
    const clock = document.getElementById('clock');
    if (clock) clock.textContent = new Date().toLocaleTimeString();
  }, 1000);

  loadModule('router');
})();
