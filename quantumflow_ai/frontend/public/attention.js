// quantumflow_ai/frontend/public/attention.js

document.getElementById("attentionForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const queryInput = document.getElementById("query").value;
  const keyInput = document.getElementById("key").value;
  const valueInput = document.getElementById("value").value;

  const useQuantumKernel = document.getElementById("useQuantumKernel").checked;
  const useVQC = document.getElementById("useVQC").checked;
  const useQAOA = document.getElementById("useQAOA").checked;
  const useQPE = document.getElementById("useQPE").checked;
  const contrastiveTest = document.getElementById("contrastiveTest").checked;

  let query, key, value;
  try {
    query = JSON.parse(queryInput);
    key = JSON.parse(keyInput);
    value = JSON.parse(valueInput);
  } catch (err) {
    alert("Invalid JSON format in one of the matrices.");
    return;
  }

  const payload = {
    query,
    key,
    value,
    use_quantum_kernel: useQuantumKernel,
    use_vqc: useVQC,
    use_qaoa_sampling: useQAOA,
    use_positional_encoding: useQPE,
    contrastive_test: contrastiveTest,
  };

  try {
    const response = await fetch("/q-attention/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    document.getElementById("attentionResult").textContent =
      JSON.stringify(result, null, 2);
  } catch (error) {
    document.getElementById("attentionResult").textContent =
      "Error running attention module: " + error;
  }
});
