// app.js

document.addEventListener("DOMContentLoaded", () => {
  const lastPage = sessionStorage.getItem("lastModule");

  // Optional auto-redirect to last used module (if needed)
  if (lastPage && location.pathname.endsWith("index.html")) {
    console.log(`Last used module: ${lastPage}`);
    // window.location.href = `/${lastPage}.html`; // enable if desired
  }

  // Save module navigation state when buttons are clicked
  document.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      let module = "home";

      const text = button.textContent.toLowerCase();
      if (text.includes("routing")) module = "router";
      else if (text.includes("energy")) module = "energy";
      else if (text.includes("compression")) module = "compression";
      else if (text.includes("decompression")) module = "decompressor";  // ✅ NEW

      sessionStorage.setItem("lastModule", module);
    });
  });
});
