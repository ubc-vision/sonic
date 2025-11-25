document.addEventListener('DOMContentLoaded', function () {
  console.log('slider.js: DOMContentLoaded');

  if (!window.juxtapose) {
    console.warn('slider.js: juxtapose library not found on window');
    return;
  }

  // Ensure any .juxtapose elements on the page are initialized.
  try {
    juxtapose.scanPage();
    console.log('slider.js: called juxtapose.scanPage()');
  } catch (err) {
    console.error('slider.js: error calling juxtapose.scanPage()', err);
  }
});
