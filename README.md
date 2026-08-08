# ETL PIPELINE FOR BILLBOARD TOP 10

Extraction: 

Using API from Billboard, pull the list of 100 artist, title , position and isNew. 

Using that artist and title using that information as dynamic parameters to get metrics from Youtube and Genius api 

PS: extraction can be orchastrated by a cron job before Billboard drops top 100 for the new week.

Transform & Clean :
Using pandas to assemble a table titled: 
song-week stats 
columns of table
1) title
2) views
3) comments 
4) likes
5) dislikes
6) popularity 
7) list of events
8) artist 
9) position
10) isNew

Load:
Connect to an mySQL database and load song-week stats