// quantumflow_ai/frontend/public/decompressor.js

document.getElementById('decompressForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('fileInput');
  const formData = new FormData();

  formData.append("file", fileInput.files[0]);
  formData.append("use_qft", document.getElementById("useQFT").checked);
  formData.append("learnable_qft", document.getElementById("learnableQFT").checked);
  formData.append("amplitude_estimate", document.getElementById("amplitudeEstimate").checked);
  formData.append("use_hhl", document.getElementById("useHHL").checked);
  formData.append("use_lstm", document.getElementById("useLSTM").checked);
  formData.append("use_ae", document.getElementById("useAE").checked);
  formData.append("alpha", document.getElementById("alpha").value);

  const resultBox = document.getElementById("resultBox");
  const resultText = document.getElementById("resultText");
  resultBox.classList.add("hidden");

  try {
    const response = await fetch("/q-decompress", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Unknown error during decompression.");
    }

    const data = await response.json();
    const output = {
      decoded: data.decoded,
      features: data.features,
      ...(data.refined ? { refined: data.refined } : {}),
      ...(data.amplitudes ? { amplitudes: data.amplitudes } : {}),
    };

    resultText.textContent = JSON.stringify(output, null, 2);
    resultBox.classList.remove("hidden");
  } catch (err) {
    resultText.textContent = "❌ Error: " + err.message;
    resultBox.classList.remove("hidden");
  }
});
