window.modules = window.modules || {};
window.modules.energy = {
  init(container) {
    const form = container.querySelector('#energy-form');
    const matrixEl = container.querySelector('#energy-matrix');
    const quantumEl = container.querySelector('#energy-quantum');
    const mlEl = container.querySelector('#energy-ml');
    const resultEl = container.querySelector('#energy-result');
    const canvas = container.querySelector('#energy-canvas');
    const ctx = canvas.getContext('2d');

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      let matrix;
      try {
        matrix = JSON.parse(matrixEl.value);
      } catch (err) {
        matrix = [];
      }
      const payload = {
        matrix,
        use_quantum: quantumEl.checked,
        use_ml: mlEl.checked,
      };
      fetch('/q-energy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
        .then(r => r.json())
        .then(data => {
          resultEl.textContent = JSON.stringify(data, null, 2);
          resultEl.classList.add('visible');
          animateCurve();
        })
        .catch(err => {
          resultEl.textContent = 'Error: ' + err.message;
          resultEl.classList.add('visible');
        });
    });

    container.querySelector('#energy-wave').addEventListener('click', animateCurve);

    function animateCurve() {
      let x = 0;
      function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        for (let i = 0; i < canvas.width; i++) {
          const y = canvas.height / 2 + Math.cos((i + x) / 15) * 30;
          ctx.lineTo(i, y);
        }
        ctx.strokeStyle = '#00cc88';
        ctx.stroke();
        x += 2;
        if (x < 2000) requestAnimationFrame(draw);
      }
      requestAnimationFrame(draw);
    }
  }
};
