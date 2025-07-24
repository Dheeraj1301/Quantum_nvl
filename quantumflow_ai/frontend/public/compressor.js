document.getElementById('compressForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('fileInput');
  const useQuantum = document.getElementById('quantumToggle').checked;
  const noise = document.getElementById('noiseToggle').checked;
  const noiseLevel = document.getElementById('noiseLevel').value;
  const useDropout = document.getElementById('dropoutToggle').checked;
  const dropoutProb = document.getElementById('dropoutProb').value;
  const denoise = document.getElementById('denoiseToggle').checked;
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
