document.addEventListener('DOMContentLoaded', function () {
    const apiUrlInput = document.getElementById('apiUrlInput');
    const dropdownButton = document.getElementById('dropdownButton');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const responsesContainer = document.getElementById('responsesContainer');
    const promptInput = document.getElementById('promptInput');
    const sendButton = document.getElementById('sendButton');
    const errorMessage = document.getElementById('errorMessage');

    const defaultApiUrl = window.location.origin + '/generate_images';

    let apiUrl = localStorage.getItem('lastUsedUrl') || defaultApiUrl;
    let urlHistory = JSON.parse(localStorage.getItem('urlHistory')) || [];

    // If urlHistory is empty, add the default localhost URL
    if (urlHistory.length === 0) {
        saveUrl(defaultApiUrl)
    }

    apiUrlInput.value = apiUrl;
    updateDropdown();

    function updateDropdown() {
        dropdownMenu.innerHTML = urlHistory.length
            ? urlHistory.map(url => `<div class="dropdown-item">
                    <span class="select-url">${url}</span>
                    <button class="delete-button">🗑️</button>
                </div>`).join('')
            : '<div class="dropdown-item">No history</div>';
    }

    function addMessage(sender, body, avatarUrl) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
    
        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('message-avatar');
        avatarDiv.innerHTML = avatarUrl ? `<img src="${avatarUrl}" alt="Avatar">` : `<div class="default-avatar">${sender[0]}</div>`;
    
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
    
        // Generate timestamp
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
        messageContent.innerHTML = `
            <div class="message-header">
                <span class="message-sender">${sender}</span>
                <span class="message-timestamp">${timestamp}</span>
            </div>
            <div class="message-body"></div>
        `;
    
        const messageBody = messageContent.querySelector('.message-body');
        if (typeof body === 'string') {
            messageBody.textContent = body;
        } else {
            messageBody.appendChild(body);
        }
    
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(messageContent);
        responsesContainer.appendChild(messageDiv);
        responsesContainer.scrollTop = responsesContainer.scrollHeight;
    }
    

    dropdownButton.addEventListener('click', () => {
        dropdownMenu.style.display = dropdownMenu.style.display === 'none' ? 'block' : 'none';
    });

    dropdownMenu.addEventListener('click', (e) => {
        if (e.target.classList.contains('select-url')) {
            apiUrl = e.target.textContent;
            apiUrlInput.value = apiUrl;
            saveUrl(apiUrl);
            dropdownMenu.style.display = 'none';
        } else if (e.target.classList.contains('delete-button')) {
            const parent = e.target.closest('.dropdown-item');
            const urlToDelete = parent.querySelector('.select-url').textContent;
            urlHistory = urlHistory.filter(url => url !== urlToDelete);
            localStorage.setItem('urlHistory', JSON.stringify(urlHistory));
            updateDropdown();
        }
    });

    apiUrlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            apiUrl = apiUrlInput.value.trim();  // Ensure the latest value is saved
            saveUrl(apiUrl);
        }
    });

    function saveUrl(url) {
        if (!url.trim()) return;

        // Set the last used URL
        localStorage.setItem('lastUsedUrl', url);

        // Add to history if it doesn't exist
        if (!urlHistory.includes(url)) {
            urlHistory.push(url);
            localStorage.setItem('urlHistory', JSON.stringify(urlHistory));
            updateDropdown();
        }

        addMessage('System', `API URL set to: ${url}`, './system.png');
    }
    async function generateImage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage('You', prompt, './user.png');
        promptInput.value = '';
        sendButton.disabled = true;
        sendButton.textContent = 'Generating...';
        errorMessage.style.display = 'none';

        try {
            const params = new URLSearchParams();
            params.set('prompt', prompt);
            params.set('seed', Math.floor(Math.random() * 1000000).toString());
            params.set('guidance', '5');

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: params.toString(),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to generate image. Status: ${response.status}, Response: ${errorText}`);
            }

            const blob = await response.blob();
            const img = document.createElement('img');
            img.src = URL.createObjectURL(blob);
            img.style.maxWidth = '100%';
            img.style.borderRadius = '8px';

            addMessage('AI', img, './child.jpg');
        } catch (err) {
            addMessage('System', `Error generating image: ${err.message}`, './system.png');
            errorMessage.textContent = `Error generating image: ${err.message}`;
            errorMessage.style.display = 'block';
        } finally {
            sendButton.disabled = false;
            sendButton.textContent = 'Send';
        }
    }

    sendButton.addEventListener('click', generateImage);

    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generateImage();
        }
    });
});