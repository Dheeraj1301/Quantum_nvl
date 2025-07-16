
// app.js

// Example: Redirect user to last opened module
document.addEventListener("DOMContentLoaded", () => {
  const lastPage = sessionStorage.getItem("lastModule");
  if (lastPage && location.pathname.endsWith("index.html")) {
    // Optionally auto-redirect or highlight
    console.log(`Last used module: ${lastPage}`);
  }

  // Optional: Save last navigation
  document.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      const module = button.textContent.includes("Routing") ? "router" : "energy";
      sessionStorage.setItem("lastModule", module);
    });
  });
});
