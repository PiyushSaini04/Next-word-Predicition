// Configuration
const API_URL = 'http://localhost:5000/predict';
const DEBOUNCE_DELAY = 300; // milliseconds

// State
let currentSuggestions = [];
let selectedIndex = -1;
let debounceTimer = null;

// DOM elements
const textInput = document.getElementById('textInput');
const suggestionsContainer = document.getElementById('suggestions');

// Initialize
textInput.addEventListener('input', handleInput);
textInput.addEventListener('keydown', handleKeyDown);
textInput.addEventListener('blur', () => {
    // Delay hiding suggestions to allow clicks
    setTimeout(() => {
        if (!suggestionsContainer.matches(':hover')) {
            clearSuggestions();
        }
    }, 200);
});

// Handle text input with debouncing
function handleInput(event) {
    const text = event.target.value.trim();
    
    // Clear previous timer
    clearTimeout(debounceTimer);
    
    // Clear suggestions if input is empty
    if (!text) {
        clearSuggestions();
        return;
    }
    
    // Debounce API calls
    debounceTimer = setTimeout(() => {
        fetchPredictions(text);
    }, DEBOUNCE_DELAY);
}

// Handle keyboard events
function handleKeyDown(event) {
    // Tab key - accept first suggestion
    if (event.key === 'Tab' && currentSuggestions.length > 0) {
        event.preventDefault();
        acceptSuggestion(0);
        return;
    }
    
    // Arrow keys - navigate suggestions
    if (event.key === 'ArrowDown') {
        event.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, currentSuggestions.length - 1);
        updateHighlight();
        return;
    }
    
    if (event.key === 'ArrowUp') {
        event.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, -1);
        updateHighlight();
        return;
    }
    
    // Enter key - accept highlighted suggestion
    if (event.key === 'Enter' && selectedIndex >= 0 && selectedIndex < currentSuggestions.length) {
        event.preventDefault();
        acceptSuggestion(selectedIndex);
        return;
    }
    
    // Escape key - clear suggestions
    if (event.key === 'Escape') {
        clearSuggestions();
        return;
    }
}

// Fetch predictions from API
async function fetchPredictions(text) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Map the backend response to plain words
        const predictions = data.suggestions.map(s => s.word);
        displaySuggestions(predictions);

    } catch (error) {
        console.error('Error fetching predictions:', error);
        suggestionsContainer.innerHTML = '<div class="suggestion" style="color: #dc3545;">Error loading suggestions</div>';
    }
}

// Display suggestions
function displaySuggestions(predictions) {
    currentSuggestions = predictions;
    selectedIndex = -1;
    
    if (predictions.length === 0) {
        suggestionsContainer.innerHTML = '';
        return;
    }
    
    suggestionsContainer.innerHTML = predictions.map((word, index) => {
        return `<div class="suggestion" data-index="${index}">${escapeHtml(word)}</div>`;
    }).join('');
    
    // Add click handlers
    suggestionsContainer.querySelectorAll('.suggestion').forEach((suggestion, index) => {
        suggestion.addEventListener('click', () => acceptSuggestion(index));
    });
}

// Accept a suggestion
function acceptSuggestion(index) {
    if (index < 0 || index >= currentSuggestions.length) {
        return;
    }
    
    const word = currentSuggestions[index];
    const cursorPos = textInput.selectionStart;
    const text = textInput.value;
    
    // Insert word at cursor position
    const before = text.substring(0, cursorPos);
    const after = text.substring(cursorPos);
    
    // Add space before word if needed
    const spaceBefore = (before.length > 0 && !before.endsWith(' ')) ? ' ' : '';
    
    // Add space after word
    const newText = before + spaceBefore + word + ' ' + after;
    
    textInput.value = newText;
    
    // Set cursor position after inserted word
    const newCursorPos = cursorPos + spaceBefore.length + word.length + 1;
    textInput.setSelectionRange(newCursorPos, newCursorPos);
    
    // Clear suggestions
    clearSuggestions();
    
    // Trigger input event to get new predictions
    textInput.dispatchEvent(new Event('input'));
}

// Update highlight for arrow key navigation
function updateHighlight() {
    suggestionsContainer.querySelectorAll('.suggestion').forEach((suggestion, index) => {
        if (index === selectedIndex) {
            suggestion.classList.add('highlighted');
        } else {
            suggestion.classList.remove('highlighted');
        }
    });
}

// Clear suggestions
function clearSuggestions() {
    currentSuggestions = [];
    selectedIndex = -1;
    suggestionsContainer.innerHTML = '';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


