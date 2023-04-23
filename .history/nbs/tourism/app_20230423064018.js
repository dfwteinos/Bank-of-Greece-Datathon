// Initialize the map
const map = L.map('map').setView([40.416775, -3.703790], 6);

// Add the base layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Create the layers for the different options
const option1Layer = L.geoJSON(option1Data, {
    style: { color: 'blue' }
});

const option2Layer = L.geoJSON(option2Data, {
    style: { color: 'red' }
});

// Function to handle the map option change
function changeMap(option) {
    if (option === 'option1') {
        option2Layer.removeFrom(map);
        option1Layer.addTo(map);
    } else {
        option1Layer.removeFrom(map);
        option2Layer.addTo(map);
    }
}

// Add event listeners to the map options
document.getElementById('option1').addEventListener('mouseover', () => changeMap('option1'));
document.getElementById('option2').addEventListener('mouseover', () => changeMap('option2'));