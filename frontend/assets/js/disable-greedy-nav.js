// Disable greedy navigation completely
(function() {
    'use strict';
    
    console.log('Disabling greedy navigation...');
    
    // Override the greedy navigation function
    window.updateNav = function() {
        // Do nothing - prevent navigation hiding
        console.log('updateNav called but disabled');
    };
    
    // When DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM loaded, forcing navigation visibility...');
        
        // Force all navigation items to be visible
        var visibleLinks = document.querySelector('#site-nav .visible-links');
        var hiddenLinks = document.querySelector('#site-nav .hidden-links');
        var button = document.querySelector('#site-nav button');
        
        if (visibleLinks) {
            visibleLinks.style.display = 'flex';
            visibleLinks.style.visibility = 'visible';
            visibleLinks.style.opacity = '1';
            console.log('Forced visible-links to show');
        }
        
        if (hiddenLinks) {
            hiddenLinks.style.display = 'none';
            console.log('Forced hidden-links to hide');
        }
        
        if (button) {
            button.style.display = 'none';
            console.log('Forced button to hide');
        }
        
        // Move any items from hidden back to visible
        if (hiddenLinks && visibleLinks) {
            var hiddenItems = hiddenLinks.children;
            while (hiddenItems.length > 0) {
                visibleLinks.appendChild(hiddenItems[0]);
                console.log('Moved item back to visible');
            }
        }
    });
    
    // Also run after window loads
    window.addEventListener('load', function() {
        console.log('Window loaded, ensuring navigation stays visible...');
        
        var visibleLinks = document.querySelector('#site-nav .visible-links');
        var hiddenLinks = document.querySelector('#site-nav .hidden-links');
        var button = document.querySelector('#site-nav button');
        
        if (visibleLinks) {
            visibleLinks.style.display = 'flex !important';
            visibleLinks.style.visibility = 'visible !important';
            visibleLinks.style.opacity = '1 !important';
        }
        
        if (hiddenLinks) {
            hiddenLinks.style.display = 'none !important';
        }
        
        if (button) {
            button.style.display = 'none !important';
        }
    });
    
    // Disable resize events that trigger greedy nav
    window.addEventListener('resize', function(e) {
        e.stopImmediatePropagation();
        console.log('Resize event blocked');
    }, true);
    
})();