// quantumflow_ai/frontend/public/app.js

document.addEventListener("DOMContentLoaded", () => {
  const lastPage = sessionStorage.getItem("lastModule");

  // Optional: auto-redirect to last used module
  if (lastPage && location.pathname.endsWith("index.html")) {
    console.log(`Last used module: ${lastPage}`);
    // window.location.href = `/${lastPage}.html`; // enable if desired
  }

  // Save module navigation state
  document.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      let module = "home";
      const text = button.textContent.toLowerCase();

      if (text.includes("routing")) module = "router";
      else if (text.includes("energy")) module = "energy";
      else if (text.includes("compression")) module = "compressor";
      else if (text.includes("decompression")) module = "decompressor";
      else if (text.includes("hpo")) module = "hpo";
      else if (text.includes("nvlink")) module = "nvlinkopt"; // ✅ NEW

      sessionStorage.setItem("lastModule", module);
    });
  });
});
