## Transition based Bottom-up RST-style Text-level Discourse Parser

<b>-- General Information</b>
```
   1. This RST-style discourse parser produces discourse tree structure on full-text level, given a raw text.
   2. This work explores the interal node representation learning with respect to tree depth. As shown in Figure 3, 
   this work finds that our model prefers state transition information and the principle component of text spans when 
   EDU number of text span grows larger (a higher tree). 
   As shown in Figure 4, this work also finds that a better representation (with automatic information flow 
   incorporation), the proposed parser obtains better performance for upper-layer tree nodes.
```
![Image text](https://github.com/NLP-Discourse-SoochowU/rst_dp2019/blob/master/data/img/fg.png)


<b>-- Required Packages</b>
```
   torch==0.4.0 
   numpy==1.14.1
   nltk==3.3
   stanfordcorenlp==3.9.1.1
```

<b>-- Training Your Own RST Parser</b>
```
    Run main.py

```

<b>-- RST Parsing with Raw Documents</b>
```
   1. Prepare your raw documents in data/raw_txt in the format of *.out
   2. Run the Stanford CoreNLP with the given bash script corpus_rst.sh using the command "./corpus_rst.sh "
   3. Run parser.py to parse these raw documents into objects of rst_tree class (Wrap them into trees).
      - segmentation
      - wrap them into trees, saved in "data/trees_parsed/trees_list.pkl"
   4. Run drawer.py to draw those trees out by NLTK
   Note: We did not provide parser codes and it can be easily implemented referring to our previous project.
```
[rst_dp2018](https://github.com/NLP-Discourse-SoochowU/rst_dp2018)

<b>-- Reference</b>

   Please read the following paper for more technical details
   
   [Longyin Zhang, Xin Tan, Fang Kong and Guodong Zhou, A Recursive Information Flow Gated Model for RST-Style Text-Level Discourse Parsing.](http://tcci.ccf.org.cn/conference/2019/papers/119.pdf)

<b>-- Developer</b>
```
  Longyin Zhang
  Natural Language Processing Lab, School of Computer Science and Technology, Soochow University, China
  mail to: zzlynx@outlook.com, lyzhang9@stu.suda.edu.cn

```

MIT License

Copyright (c) 2020 Discourse Analysis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
