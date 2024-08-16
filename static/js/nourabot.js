let sessionId = null;

// Function to generate session ID
function generateSessionId() {
    const requestOptions = {
        method: "GET",
        redirect: "follow"
    };

    return fetch("https://testapi.unomiru.com/api/Waysbot/generate_sessionid", requestOptions)
        .then((response) => response.text())
        .then((result) => {
            console.log("Generated Session ID:", result);
            sessionId = result;
        })
        .catch((error) => {
            console.error("Error generating session ID:", error);
        });
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    const chatHistory = document.getElementById('chat-history');
    const loadingSpinner = document.getElementById('loading-spinner');

    if (message !== '') {
        const chatMessageContainer = document.createElement('div');
        chatMessageContainer.classList.add('user-chat-message-container');

        const userIcon = document.createElement('div');
        userIcon.classList.add('user-icon');
        userIcon.innerHTML = '<img src="../static/Assets/images/user-icon.png" alt="User Icon">';

        const userMessage = document.createElement('div');
        userMessage.classList.add('chat-message', 'user-message');
        userMessage.textContent = message;

        chatMessageContainer.appendChild(userMessage);
        chatMessageContainer.appendChild(userIcon);
        chatHistory.prepend(chatMessageContainer);

        loadingSpinner.style.display = 'block';

        if (!sessionId) {
            generateSessionId().then(() => {
                if (sessionId) {
                    sendMessageToApi(message);
                }
            });
        } else {
            sendMessageToApi(message);
        }

        input.value = ''; // Clear input
    }
}

function sendMessageToApi(message) {
    const chatHistory = document.getElementById('chat-history');
    const loadingSpinner = document.getElementById('loading-spinner');

    const requestOptions = {
        method: "POST",
        headers: {
            "Authorization": "Adxbght54723xtyiyqqtuv",
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "user_input": message, "session_id": sessionId }),
        redirect: "follow"
    };

    fetch("https://api.retailme.waysaheadglobal.com/api/ask", requestOptions)
        .then(response => response.json())
        .then(result => {
            const apiResponse = result.answer || result.response || result.text;

            const chatMessageContainer = document.createElement('div');
            chatMessageContainer.classList.add('noura-chat-message-container');

            const nouraIcon = document.createElement('div');
            nouraIcon.classList.add('noura-icon');
            nouraIcon.innerHTML = '<img src="../static/Assets/images/noura-icon.png" alt="Noura Icon">';

            const responseMessage = document.createElement('div');
            responseMessage.classList.add('chat-message', 'noura-message');
            responseMessage.textContent = apiResponse;

            chatMessageContainer.appendChild(nouraIcon);
            chatMessageContainer.appendChild(responseMessage);
            chatHistory.prepend(chatMessageContainer);

            loadingSpinner.style.display = 'none';

            // Trigger voice response
            if (!isMuted) {
                speakMessage(apiResponse);
            }
        })
        .catch(error => {
            console.error("Error:", error);
            loadingSpinner.style.display = 'none';
        });
}


// Handle Enter key
function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

let isMuted = false;
let ongoingSpeech = null;  // Track the current speech synthesis instance
let lastMessage = '';      // Track the last spoken message

function toggleSound() {
    isMuted = !isMuted;
    const soundControl = document.querySelector('.sound-control');
    const nouraAvatar = document.getElementById('noura-avatar');
    const listeningAvatar = nouraAvatar.querySelector('img[alt="Listening Avatar"]');
    const talkingAvatar = nouraAvatar.querySelector('img[alt="Talking Avatar"]');

    soundControl.textContent = isMuted ? '🔇' : '🔊';

    if (isMuted) {
        // If muted, cancel any ongoing speech and clear the last message
        speechSynthesis.cancel();
        talkingAvatar.classList.remove('active');
        listeningAvatar.classList.add('active');
        lastMessage = ''; // Clear the lastMessage so it won't be spoken again
    }
}

function speakMessage(message, callback) {
    // Cancel any ongoing speech when starting a new one
    speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(message);
    const voices = speechSynthesis.getVoices();

    let desiredVoice = voices.find(voice =>
        voice.name.toLowerCase().includes('female') ||
        voice.name.toLowerCase().includes('woman') ||
        voice.name.toLowerCase().includes('samantha') ||
        voice.name.toLowerCase().includes('victoria') ||
        voice.name.toLowerCase().includes('zira') ||
        voice.name.toLowerCase().includes('joanna') ||
        voice.name.toLowerCase().includes('tessa')
    );

    speech.voice = desiredVoice || voices[0];

    const nouraAvatar = document.getElementById('noura-avatar');
    const listeningAvatar = nouraAvatar.querySelector('img[alt="Listening Avatar"]');
    const talkingAvatar = nouraAvatar.querySelector('img[alt="Talking Avatar"]');

    speech.onend = () => {
        ongoingSpeech = null;
        // Switch back to listening avatar when speech ends
        talkingAvatar.classList.remove('active');
        listeningAvatar.classList.add('active');
        if (callback) callback();
    };

    ongoingSpeech = speech;
    lastMessage = message;  // Store the last message

    if (!isMuted) {
        // If not muted, switch to talking avatar and speak the message
        listeningAvatar.classList.remove('active');
        talkingAvatar.classList.add('active');
        speechSynthesis.speak(speech);
    } else {
        // If muted, switch to listening avatar immediately
        talkingAvatar.classList.remove('active');
        listeningAvatar.classList.add('active');
        if (callback) callback();
    }
}

function toggleChat() {
    const chatbot = document.getElementById('chatbot');
    chatbot.classList.toggle('expanded');

    // If the chat window is being closed, stop any ongoing speech and clear the lastMessage
    if (!chatbot.classList.contains('expanded')) {
        speechSynthesis.cancel(); // Stop ongoing speech
        lastMessage = ''; // Clear the last spoken message
        ongoingSpeech = null; // Reset the ongoing speech instance
    }
}

// Mock function for microphone button
function startListening() {
    alert('Microphone feature not implemented yet.');
}

speechSynthesis.onvoiceschanged = function () {
    console.log('Available voices:', speechSynthesis.getVoices());
};

// Check if the browser supports the SpeechRecognition API
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onstart = function () {
        console.log("Voice recognition started. Try speaking into the microphone.");
        document.getElementById('mic-button').classList.add('listening');
    };

    recognition.onspeechend = function () {
        console.log("Voice recognition ended.");
        document.getElementById('mic-button').classList.remove('listening');
        recognition.stop();
    };

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        console.log("You said: " + transcript);
        document.getElementById('chat-input').value = transcript;
        sendMessage(); // Automatically send the message after speech recognition
    };

    recognition.onerror = function (event) {
        console.error("Error occurred in recognition: " + event.error);
    };

    function startListening() {
        recognition.start();
    }
} else {
    console.log("Speech recognition not supported in this browser.");
    alert("Sorry, your browser doesn't support speech recognition.");
}
