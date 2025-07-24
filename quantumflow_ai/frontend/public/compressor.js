document.getElementById('compressForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('fileInput');

  // Gracefully handle optional toggles that may not exist yet
  const useQuantumEl = document.getElementById('quantumToggle');
  const useQuantum = useQuantumEl ? useQuantumEl.checked : false;

  const noiseEl = document.getElementById('noiseToggle');
  const noise = noiseEl ? noiseEl.checked : false;

  const noiseLevelEl = document.getElementById('noiseLevel');
  const noiseLevel = noiseLevelEl ? noiseLevelEl.value : 0;

  const denoiseEl = document.getElementById('denoiseToggle');
  const denoise = denoiseEl ? denoiseEl.checked : false;

  const output = document.getElementById('output');

  if (!fileInput.files.length) {
    output.textContent = "Please select a CSV file.";
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  const queryParams = new URLSearchParams({
    use_quantum: useQuantum,
    use_denoiser: denoise,
    noise: noise,
    noise_level: noiseLevel,
    use_dropout: useDropout,
    dropout_prob: dropoutProb,
  });

  try {
    const response = await fetch(`/q-compression/upload?${queryParams.toString()}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const result = await response.json();
    output.textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    output.textContent = `Error: ${err.message}`;
  }
});
