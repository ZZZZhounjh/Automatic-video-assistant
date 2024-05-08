from langchain.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=NaaSpRMBHjg", add_video_info=False)
transcript = loader.load()
print(transcript)

with open("transcript.txt", "w", encoding="utf-8") as file:
    transcript_text = '\n'.join([document.page_content for document in transcript])
    file.write(transcript_text)