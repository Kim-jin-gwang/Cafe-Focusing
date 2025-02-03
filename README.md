# Cafe-Focusing

### date: 2021.10.18~2021.12.18

### collaborator: @malangcongdduck

🕋 Using python and opencv from Jupyter Lab

## Overcome the one-focusing limit of the primary camera ☕📷

The typical smartphone Galaxy and iPhone support food and portrait filters. All of these filters focus on the circular shape of the object. However, there is a limitation that fixed circle-shaped focusing support cannot be focused on cafe drinks or bread-shaped ones, so our team worked on this project to implement the function of finding the outline of the object and making the outline range out of focus.


For maximum lightness and ease of use, the project was implemented using technologies such as edge detection, edge contour detection and area alignment, and Gaussian blur for the purpose of implementing the project using only openv function, not artificial intelligence's area division technology.
For edge detection, we used Canny Edge detection, Dilate operation, and Erode operation, and for edge detection, we created the Contour Mask corresponding to the largest contour extracted using findContours(). After smoothing the mask, Gaussian blur and morphology techniques were used for blur processing, and then the extracted mask was made into three dimensions to extract the object and extract the background. Finally, the background blur processing and the two images were combined to obtain the final result.



** There is a limitation that food photos that are not clearly defined by the stained contour cannot be detected due to the use of artificial intelligence's area division technology, but we solved the problem by narrowing the goal of making it light and changing it to Food→Cafe.
