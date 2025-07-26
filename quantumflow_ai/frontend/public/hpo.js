// quantumflow_ai/frontend/public/scripts/hpo.js

document.getElementById("hpoForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const getEl = (id) => document.getElementById(id);
  const parseNum = (el) => el && el.value ? parseFloat(el.value) : undefined;

  // Checkboxes
  const useHybrid = getEl("useHybrid").checked;
  const useLSTM = getEl("useLSTM").checked;
  const useWarmStart = getEl("useWarmStart").checked;
  const useKernel = getEl("useKernelDecoder").checked;
  const useGumbel = getEl("useGumbel").checked;

  // Warm start model name (optional)
  const warmStartModel = useWarmStart ? getEl("modelName").value.trim() : null;

  const payload = {
    use_hybrid: useHybrid && !useLSTM && !useKernel,
    use_meta_lstm: useLSTM,
    use_kernel_decoder: useKernel,
    use_gumbel: useGumbel,
    use_warm_start: warmStartModel || null,
    hardware_context: {
      gpu_type_id: parseNum(getEl("gpuType")),
      model_size_mb: parseNum(getEl("modelSize")),
      seq_len: parseNum(getEl("seqLen")),
    },
  };

  try {
    const res = await fetch("/q-hpo/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await res.json();
    document.getElementById("hpoResult").innerText = JSON.stringify(json, null, 2);
  } catch (err) {
    document.getElementById("hpoResult").innerText = `Error: ${err.message}`;
  }
});
