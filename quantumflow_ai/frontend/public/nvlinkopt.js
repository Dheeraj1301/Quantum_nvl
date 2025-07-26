// quantumflow_ai/frontend/public/nvlinkopt.js

document.getElementById("nvlinkForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("fileInput");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const useQGNN = document.getElementById("useQGNN").checked;
  const useKernel = document.getElementById("useKernel").checked;
  const useVQAOA = document.getElementById("useVQAOA").checked;
  const useRL = document.getElementById("useRL").checked;
  const useClassifier = document.getElementById("useClassifier").checked;

  const params = new URLSearchParams({
    use_qgnn: useQGNN,
    use_kernel: useKernel,
    use_vqaoa: useVQAOA,
    use_rl: useRL,
    use_classifier: useClassifier,
  });

  try {
    const response = await fetch(`/q-nvlinkopt/run?${params.toString()}`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    document.getElementById("output").textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    document.getElementById("output").textContent = "Error: " + err.message;
  }
});
