document.addEventListener('DOMContentLoaded', () => {
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');

    // Tab switching functionality
    tabLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            // Remove active class from all links and contents
            tabLinks.forEach(l => l.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked link
            link.classList.add('active');

            // Show corresponding content
            const tabId = link.getAttribute('data-tab');
            const content = document.getElementById(tabId);
            if (content) {
                content.classList.add('active');
            }
        });
    });

    // NEW button click handlers
    const btnNewButtons = document.querySelectorAll('.btn-new');
    btnNewButtons.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            // Get the parent tab content to determine which section
            const parentTab = btn.closest('.tab-content');
            if (parentTab) {
                const tabId = parentTab.id;
                
                // Redirect based on tab
                switch(tabId) {
                    case 'receipts':
                        window.location.href = 'new_receipt.html';
                        break;
                    case 'delivery':
                        window.location.href = 'new_delivery.html';
                        break;
                    case 'adjustment':
                        window.location.href = 'new_adjustment.html';
                        break;
                    case 'transfer':
                        window.location.href = 'new_transfer.html';
                        break;
                }
            }
        });
    });
});
