
// 🧠 Listen for routing form submit
document.getElementById("routingForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const expertIds = document.getElementById("experts").value.split(",").map(Number);
  const tokenIds = document.getElementById("tokens").value.split(",").map(Number);
  const useQuantum = document.getElementById("useQuantum").checked;

  const model_graph = { experts: expertIds };
  const token_stream = tokenIds.map(id => ({ token_id: id }));

  fetch("/q-routing", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_graph, token_stream, use_quantum: useQuantum })
  })
    .then(res => {
      if (!res.ok) throw new Error("Routing failed.");
      return res.json();
    })
    .then(data => {
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML = `
        <h3>Routing Results</h3>
        <p><strong>Status:</strong> ${data.status}</p>
        <p><strong>Mode:</strong> ${data.mode}</p>
        <p><strong>Routing Score:</strong> ${data.routing_score}</p>
        <pre><strong>Params:</strong> ${JSON.stringify(data.optimized_params, null, 2)}</pre>
      `;

      const assignments = data.results.assignments || data.results;
      if (!Array.isArray(assignments)) {
        outputDiv.innerHTML += `<p style="color:red;">No routing assignments found to plot.</p>`;
        return;
      }

      analyzeRouting(assignments, expertIds.length);
    })
    .catch(err => {
      document.getElementById("output").innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    });
});

// 📂 Handle CSV file upload
document.getElementById("fileForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("csvFile");
  const useQuantum = document.getElementById("useQuantumFile").checked;

  if (!fileInput.files.length) {
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("use_quantum", useQuantum);

  fetch("/q-routing/file-upload", {
    method: "POST",
    body: formData
  })
    .then(res => {
      if (!res.ok) throw new Error("Routing failed.");
      return res.json();
    })
    .then(data => {
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML = `
        <h3>Routing Results</h3>
        <p><strong>Status:</strong> ${data.status}</p>
        <p><strong>Mode:</strong> ${data.mode}</p>
        <p><strong>Routing Score:</strong> ${data.routing_score}</p>
        <pre><strong>Params:</strong> ${JSON.stringify(data.optimized_params, null, 2)}</pre>
      `;

      const assignments = data.results.assignments || data.results;
      if (!Array.isArray(assignments)) {
        outputDiv.innerHTML += `<p style="color:red;">No routing assignments found to plot.</p>`;
        return;
      }

      const expertCount = assignments.reduce((m, a) => Math.max(m, a.expert), -1) + 1;
      analyzeRouting(assignments, expertCount);
    })
    .catch(err => {
      document.getElementById("output").innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    });
});


// 📊 Print analysis summary
function analyzeRouting(assignments, numExperts) {
  const expertCount = Array(numExperts).fill(0);
  assignments.forEach(a => expertCount[a.expert]++);

  const maxLoad = Math.max(...expertCount);
  const minLoad = Math.min(...expertCount);
  const avg = (expertCount.reduce((a, b) => a + b, 0) / numExperts).toFixed(2);

  const overloaded = expertCount.map((c, i) => c > avg * 1.5 ? i : null).filter(x => x !== null);
  const underused = expertCount.map((c, i) => c < avg * 0.5 ? i : null).filter(x => x !== null);

  const insights = `
    <h4>📊 Routing Analysis</h4>
    <ul>
      <li><strong>Max Load:</strong> ${maxLoad}</li>
      <li><strong>Min Load:</strong> ${minLoad}</li>
      <li><strong>Average Load:</strong> ${avg}</li>
      <li><strong>Overloaded Experts:</strong> ${overloaded.length > 0 ? overloaded.join(", ") : "None"}</li>
      <li><strong>Underused Experts:</strong> ${underused.length > 0 ? underused.join(", ") : "None"}</li>
    </ul>
  `;

  document.getElementById("output").innerHTML += insights;
}
