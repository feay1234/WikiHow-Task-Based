HowToKB data:
==========================================================================================================================================
  - wikihow-data-all.json: wikihow data in .json format
    + Number of documents: 168697
    + Format: attributes of each document:
      - title: String
      - Question explanation: String
      - link: String (id of string - looking at file wikihow-id-url)
      - Tips: String
      - Warnings: String
      - Video: String
      - Things you will need: List of String
      - Ingredients: List of String
      - Views: integer
      - rate: double
      - Category: id of category (looking at file wikihow-id-category)
      - Answer: List of methods, each method includes list of parts. Each part includes list of steps. Each each step include:
	+ Order: integer
	+ Heading: String
	+ Detail: String
	+ Image: String
	
==========================================================================================================================================
  - task-frame-before-clustering.json: task frames extracted from wikihow data before clustering in .json format
    + Number of task frames: 1292250
    + Format: Attributes of each task frames:
      - id: integer
      - locations: list of strings
      - temporal: list of strings
      - participating object: list of strings
      - participating living being: list of strings
      - task:
	+ id: integer
	+ head verb: string
	+ head noun: string
	+ original verb: string
	+ original noun: string
	+ url_image: string
	+ category id: integer
	+ link id: integer
	+ rating: double
	+ view: integer
	+ video: if yes
	+ parent: task frame id
	+ prev: task frame id
	+ next: task frame id
	+ sub-task: list of task frame id
	
==========================================================================================================================================	
  - task-frame-after-clustering.json: task frames after clustering: merging all task frames in a same cluster into an task frame
    + Number of task frames: 512537
    + Format: similar to task frame before clustering
    
==========================================================================================================================================
  - task-frame-id-before-to-task-frame-id-after: mapping task frame id before clustering to after clustering
  
  
  - wikihow-id-category.txt: mapping id to category of wikihow
  
  
  - wikihow-id-url: mapping id to url of wikihow
  
  
==========================================================================================================================================
==========================================================================================================================================
Data for usecase:

  - task-frame-have-video.json: wikihow task frames which have youtube video (extract from wikihow data)
  
  
  - uniq-youtube-video.json: youtube video information crawled from youtube in .json format
  
  - id-frame-wikihow-url-youtube-link: mapping: taskframe id -- wikihow url -- youtube link
  
  
  
  
  
  
