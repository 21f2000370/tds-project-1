# TDS Project 1

## I have used the Pyhton code to scrape the github data using github API

- First, I generated a personal access token on GitHub.
- Then, I wrote a Python script that begins by calling GitHub’s user search API, passing in location and follower count as parameters. The API request is paginated using a loop, where each iteration retrieves 100 users until there are fewer than 100 users returned, at which point all user data is stored in a users list.
- After collecting the initial user data, I loop through the users list and call the user API to fetch additional details for each user.
- During this iteration, I also compiled the specific fields needed for each user’s data.
- In the same loop, I called the repository API to get a list of each user’s public repositories and prepared the relevant fields for the repository data.
- Finally, I added both the user and repository data to a pandas DataFrame to save everything into a CSV file.

## Huggingface has noticeably higher followers and also for hireable true users/company followers numbers are high

## Developers should follow company where hireable is true


