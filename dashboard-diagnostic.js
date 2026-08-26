// Dashboard Diagnostic Script
// Run this in browser console to check for issues

console.log('=== Dashboard Diagnostic ===');

// 1. Check if required elements exist
const elements = {
    waveBody: document.getElementById('waveBody'),
    regimePane: document.querySelector('.top-pane[data-top="regime"]'),
    wavePane: document.querySelector('.top-pane[data-top="wave"]'),
    techPane: document.querySelector('.top-pane[data-top="tech"]'),
    elnPane: document.querySelector('.top-pane[data-top="eln"]'),
    earningsPane: document.querySelector('.top-pane[data-top="earnings"]'),
    regimeTab: document.querySelector('.mega-tab[data-top="regime"]'),
    waveTab: document.querySelector('.mega-tab[data-top="wave"]'),
    techTab: document.querySelector('.mega-tab[data-top="tech"]'),
    elnTab: document.querySelector('.mega-tab[data-top="eln"]'),
    earningsTab: document.querySelector('.mega-tab[data-top="earnings"]')
};

console.log('=== Element Existence ===');
Object.entries(elements).forEach(([key, el]) => {
    console.log(`${key}: ${el ? 'EXISTS' : 'MISSING'}${el ? ' (display: ' + getComputedStyle(el).display + ')' : ''}`);
});

// 2. Check active states
console.log('\n=== Active States ===');
const activePanes = document.querySelectorAll('.top-pane.active');
const activeTabs = document.querySelectorAll('.mega-tab.active');
const activeSubViews = document.querySelectorAll('.subview.active');

console.log(`Active panes: ${activePanes.length}`);
console.log(`Active tabs: ${activeTabs.length}`);
console.log(`Active subviews: ${activeSubViews.length}`);

// 3. Check current view
console.log('\n=== Current View ===');
function curView() {
    const a = document.querySelector('.mega-leaf.active');
    if (a) return { top: a.dataset.top, sub: a.dataset.sub, leaf: a.dataset.leaf || null };
    const t = document.querySelector('.mega-tab.active');
    return { top: t ? t.dataset.top : 'regime', sub: null, leaf: null };
}

const currentView = curView();
console.log('Current view:', currentView);

// 4. Check data availability
console.log('\n=== Data Availability ===');
const lastData = window.lastData;
console.log('lastData available:', !!lastData);
console.log('waveCounts available:', lastData ? !!lastData.waveCounts : false);
console.log('waveCounts length:', lastData && lastData.waveCounts ? lastData.waveCounts.length : 'N/A');

// 5. Check wave data structure
if (lastData && lastData.waveCounts) {
    console.log('\n=== Wave Data Structure ===');
    lastData.waveCounts.forEach((wave, index) => {
        console.log(`Wave ${index + 1}:`, {
            name: wave.name,
            ticker: wave.ticker,
            currentPrice: wave.currentPrice,
            calibrated: wave.calibrated,
            hasPivots: !!wave.pivots,
            hasTargets: !!wave.currentWaveTargets
        });
    });
}

// 6. Check CSS classes
console.log('\n=== CSS Classes on Wave Pane ===');
if (elements.wavePane) {
    console.log('Wave pane classes:', Array.from(elements.wavePane.classList));
    console.log('Wave pane display:', getComputedStyle(elements.wavePane).display);
}

// 7. Test navigation
console.log('\n=== Navigation Test ===');
function testNavigation(top, sub, leaf) {
    console.log(`Testing navigation to: ${top}/${sub}/${leaf}`);

    // Check if pane exists
    const pane = document.querySelector(`.top-pane[data-top="${top}"]`);
    if (!pane) {
        console.log(`❌ Pane for ${top} not found`);
        return false;
    }

    // Check if subview exists
    const subview = pane.querySelector(`.subview[data-sub="${sub}"]`);
    if (!subview) {
        console.log(`❌ Subview for ${sub} not found`);
        return false;
    }

    console.log(`✅ Navigation target found`);
    return true;
}

// Test all valid tops
const VALID_TOPS = ['regime', 'tech', 'wave', 'eln', 'earnings'];
VALID_TOPS.forEach(top => {
    testNavigation(top, top, null);
});

console.log('\n=== Diagnostic Complete ===');