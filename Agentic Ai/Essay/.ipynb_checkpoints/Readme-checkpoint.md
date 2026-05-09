#

1. tavily - search engine for llm
2. Architecture:
   - Router - (checks whether need research or not)
   - Reasearch - (if yes then go to reserach node / if no then planning/orchestator node)
   - Planner - (it plans the actual data based on topic that we selected)
   - N Worker - (its the one who writes blog according to our application) (N according to section inside the blog)
   - reducer - combine all the sections given by N Workers - sends all combined data to one LLM and ask for where is the need of images to put, then it generates images and place where it should to be

## phases

1. Basic Blog Agent (No researc + No Images) (only textual blog)
2. add research feature (tilly)
3. image feature
4. convert it into GUI
