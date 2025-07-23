document.getElementById('compressForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('fileInput');
  const useQuantum = document.getElementById('quantumToggle').checked;
  const output = document.getElementById('output');

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const response = await fetch(`/q-compression/upload?use_quantum=${useQuantum}`, {
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
