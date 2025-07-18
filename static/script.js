// Function to handle sending the message
function sendMessage() {
  const userInput = document.getElementById('user-input').value;

  if (userInput.trim() !== "") {
    appendMessage(userInput, 'user-message');

    // Send request to Python Flask server for bot response
    fetch('/get-bot-response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ user_message: userInput })  // Send user's message to backend
    })
    .then(response => response.json())
    .then(data => {
      appendMessage(data.bot_message, 'bot-message');  // Display bot's response
    })
    .catch(error => console.error('Error:', error));

    document.getElementById('user-input').value = '';  // Clear the input field
  }
}

// Add event listener to "Send" button
document.getElementById('send-btn').addEventListener('click', sendMessage);

// Add event listener for "Enter" key press in the input field
document.getElementById('user-input').addEventListener('keydown', function(event) {
  if (event.key === 'Enter') {
    sendMessage();
  }
});

// Function to append messages to the chat window
function appendMessage(text, className) {
  const chatWindow = document.getElementById('chat-window');

  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message', className);
  messageDiv.innerText = text;

  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Handle image selection (optional feature)
document.getElementById('image-input').addEventListener('change', function (event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const imgElement = document.getElementById('display-image');
      imgElement.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});
