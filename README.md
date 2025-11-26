# SONIC: Spectral Optimization of Noise for Inpainting with Consistency

[![button](https://img.shields.io/badge/Project%20Website-orange?style=for-the-badge)](https://ubc-vision.github.io/sonic/)
[![button](https://img.shields.io/badge/Paper-blue?style=for-the-badge)](https://arxiv.org/abs/2511.19985)



<span class="author-block">
  <a href="">Seungyeon Baek</a>,
</span>
<span class="author-block">
  <a href="">Erqun Dong</a>,
</span>
<span class="author-block">
  <a href="">Shadan Namazifard</a>,
</span>
<span class="author-block">
  <a href="">Mark J. Matthews</a>,
</span>
<span class="author-block">
  <a href="https://www.cs.ubc.ca/~kmyi/">Kwang Moo Yi</a>
</span>

<hr>

## Inpainting Example
Given a nearest-pixel inpainted image (left) and its corresponding prompt, we can inpaint the image via noise optimization.

<table>
  <tr>
    <td align="center">Input</td>
    <td align="center">Inpainted Output</td>
  </tr>
  <tr>
    <td><img src="samples/FFHQ/00064.png" width="300"/></td>
    <td><img src="inpaint_output_samples/FFHQ/inpainted_00064.png" width="300"/></td>
  </tr>
</table>

```
"A young man with short black hair styled upward, dark brown eyes, and fair skin with light stubble. He has well-defined eyebrows and is wearing a black collar or shirt. The background is a clean white."
```

<hr>

## Code Release
- <input type="checkbox" checked> Inpainting code with sample images and prompts
- <input type="checkbox"> Environment setup guide
- <input type="checkbox"> Metrics code from FLAIR and BrushNet

