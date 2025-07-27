window.modules = window.modules || {};
window.modules.router = {
  init(container) {
    const form = container.querySelector('#routing-form');
    const tokensEl = container.querySelector('#route-tokens');
    const quantumEl = container.querySelector('#route-quantum');
    const resultEl = container.querySelector('#route-result');
    const heatDiv = container.querySelector('#route-heat');
    const canvas = container.querySelector('#route-canvas');
    const ctx = canvas.getContext('2d');

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      let matrix;
      try {
        matrix = JSON.parse(tokensEl.value);
      } catch (err) {
        matrix = [];
      }
      fetch('/q-routing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tokens: matrix, use_quantum: quantumEl.checked })
      })
        .then(r => r.json())
        .then(data => {
          resultEl.textContent = JSON.stringify(data, null, 2);
          resultEl.classList.add('visible');
          animateWave();
        })
        .catch(err => {
          resultEl.textContent = 'Error: ' + err.message;
          resultEl.classList.add('visible');
        });
    });

    container.querySelector('#route-heatmap').addEventListener('click', () => {
      let matrix;
      try {
        matrix = JSON.parse(tokensEl.value);
      } catch (err) {
        matrix = [];
      }
      renderHeatmap(matrix, heatDiv);
    });

    function renderHeatmap(mat, target) {
      target.textContent = '';
      const rows = mat.length;
      const cols = mat[0] ? mat[0].length : 0;
      target.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cell = document.createElement('div');
          const val = mat[r][c] || 0;
          const opacity = Math.max(0, Math.min(1, val));
          cell.style.backgroundColor = `rgba(0,255,100,${opacity})`;
          target.appendChild(cell);
        }
      }
    }

    function animateWave() {
      let t = 0;
      function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        for (let x = 0; x < canvas.width; x++) {
          const y = canvas.height / 2 + Math.sin((x + t) / 20) * 40;
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#00ff66';
        ctx.stroke();
        t += 2;
        if (t < 2000) requestAnimationFrame(draw);
      }
      requestAnimationFrame(draw);
    }
  }
};
