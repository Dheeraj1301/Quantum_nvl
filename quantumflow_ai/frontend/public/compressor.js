document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('compressForm');
  const fileInput = document.getElementById('fileInput');
  const useQuantum = document.getElementById('quantumToggle').checked;
  const output = document.getElementById('output');

  form.addEventListener('submit', async (e) => {
    e.preventDefault(); // ✅ Prevent default GET form behavior

    if (!fileInput.files.length) {
      output.textContent = "Please upload a CSV file.";
      return;
    }

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
});
