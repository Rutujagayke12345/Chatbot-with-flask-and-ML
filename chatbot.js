// static/js/chatbot.js

document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const inputField = document.querySelector("#user_input");
    const responseContainer = document.querySelector("h2");  // The area where the response will appear

    form.addEventListener("submit", function(event) {
        event.preventDefault();  // Prevent the form from refreshing the page

        const userInput = inputField.value.trim();  // Get the user input

        if (userInput) {
            // Clear the input field
            inputField.value = "";

            // Send the user input to the server via AJAX
            fetch("/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ user_input: userInput })
            })
            .then(response => response.json())
            .then(data => {
                // Display the chatbot's response
                responseContainer.innerHTML = `<span class="response">${data.response}</span>`;
            })
            .catch(error => {
                console.error('Error:', error);
                responseContainer.innerHTML = "Sorry, something went wrong!";
            });
        }
    });
});
