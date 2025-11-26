// Written by Dor Verbin, October 2021
// This is based on:
// http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    // Check if already initialized using data attribute
    if (videoMerge.dataset.initialized === 'true') {
        return;
    }

    var position = 0.5;
    var vidWidth = vid.videoWidth/2;
    var vidHeight = vid.videoHeight;
    var isTouching = false;

    var mergeContext = videoMerge.getContext("2d");

    vid.play();

    function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }

        function trackLocationTouch(e) {
            if (isTouching) {
                // Normalize to [0, 1]
                bcr = videoMerge.getBoundingClientRect();
                position = ((e.touches[0].pageX - bcr.x) / bcr.width);
            }
        }

        function startTouching(e) {
            isTouching = true;
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        function stopTouching(e) {
            isTouching = false;
        }

        videoMerge.addEventListener("mousemove", trackLocation, false);
        videoMerge.addEventListener("touchstart", startTouching, false);
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);
        videoMerge.addEventListener("touchend", stopTouching, false);

        // Mark as initialized
        videoMerge.dataset.initialized = 'true';

        function drawLoop() {
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);

            // Draw left half of video (ProPainter) on the left side
            mergeContext.drawImage(vid,
                0, 0, vidWidth, vidHeight,
                0, 0, vidWidth, vidHeight
            );
            // Draw right half of video (Our Method) on the right side, controlled by slider
            mergeContext.drawImage(vid,
                vidWidth + colStart, 0, colWidth, vidHeight,
                colStart, 0, colWidth, vidHeight
            );

            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 2;  // Center the arrow vertically
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.beginPath();
            mergeContext.arc(currX, arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340";
            mergeContext.fill();
            
            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth*position, 0);
            mergeContext.lineTo(vidWidth*position, vidHeight);
            mergeContext.closePath();
            mergeContext.strokeStyle = "#AAAAAA";
            mergeContext.lineWidth = 5;            
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth/2);
            
            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowWidth/2);
            
            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowWidth/2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowWidth/2);
            
            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowWidth/2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth/2);

            mergeContext.closePath();
            mergeContext.fillStyle = "#AAAAAA";
            mergeContext.fill();

            requestAnimationFrame(drawLoop);
        }
        
        requestAnimationFrame(drawLoop);
}

Number.prototype.clamp = function(min, max) {
    return Math.min(Math.max(this, min), max);
};

function resizeAndPlay(element) {
    var cv = document.getElementById(element.id + "Merge");

    // Clear any previous initialization flag to allow re-initialization on page reload
    cv.dataset.initialized = 'false';

    cv.width = element.videoWidth/2;
    cv.height = element.videoHeight;
    element.style.height = "0px";  // Hide video without stopping it

    // Remove loading indicator
    var loadingIndicator = element.parentElement.querySelector('.video-loading');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }

    playVids(element.id);
}