// app.js

document.addEventListener("DOMContentLoaded", () => {
  const lastPage = sessionStorage.getItem("lastModule");

  // Optional auto-redirect on fresh load of index.html
  if (lastPage && location.pathname.endsWith("index.html")) {
    console.log(`Last used module: ${lastPage}`);
    // Optional: redirect to last module
    // window.location.href = `/${lastPage}.html`;
  }

  // Save module navigation state when buttons are clicked
  document.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      let module = "home";

      if (button.textContent.toLowerCase().includes("routing")) module = "router";
      else if (button.textContent.toLowerCase().includes("energy")) module = "energy";
      else if (button.textContent.toLowerCase().includes("compression")) module = "compression";

      sessionStorage.setItem("lastModule", module);
    });
  });
});
