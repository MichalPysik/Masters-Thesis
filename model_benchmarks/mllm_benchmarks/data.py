import ffmpeg
import json

# Number of samples from the test set, used for the benchmark
# Maximum possible value is 6075
NUM_SAMPLES = 100


def read_and_parse_annotations(limit=NUM_SAMPLES):
    dataset_dir = "./dataset/SUTD-TrafficQA"
    with open(f"{dataset_dir}/R2_test.jsonl", "r") as f:
        lines = f.readlines()

    # extract legend and data
    legend = json.loads(lines[0])
    raw_data = [json.loads(line) for line in lines[1:(limit + 1)]]
    structured_data = [dict(zip(legend, row)) for row in raw_data]

    return structured_data


def sample_uniform_frames_from_video(video_path):
    # Get video duration
    probe = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])

    # For short videos under 8s, sample at 2 FPS
    sampling_fps = 2.0
    num_frames = int(duration * sampling_fps)
    # For longer videos, sample 16 frames uniformly
    if duration > 8.0:
        sampling_fps = 16.0 / duration
        num_frames = 16

    process = (
        ffmpeg.input(video_path, ss=0, t=duration)
        .output(
            "pipe:",
            format="image2pipe",
            vcodec="mjpeg",
            vf=f"fps={sampling_fps}",
            vframes=num_frames,
        )
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    out_bytes = process.stdout.read()
    process.wait()

    # Split the concatenated JPEG stream into individual frames
    frames = []
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"
    pos = 0
    while True:
        start_idx = out_bytes.find(start_marker, pos)
        if start_idx == -1:
            break
        end_idx = out_bytes.find(end_marker, start_idx)
        if end_idx == -1:
            break
        end_idx += len(end_marker)
        frames.append(out_bytes[start_idx:end_idx])
        pos = end_idx

    return frames


def describe_sutd_traffic_qa_dataset_subset(limit=NUM_SAMPLES):
    print(f"----------------SUTD Traffic QA (First {limit} samples of test subset-----------------")
    annotations = read_and_parse_annotations(limit)
    print(f"Number of questions: {len(annotations)}")

    # Some questions have less than 4 options
    avg_num_options = 0
    for annotation in annotations:
        for i in range(4):
            if annotation[f"option{i}"] != "":
                avg_num_options += 1
    avg_num_options /= len(annotations)
    print(f"Average number of valid options per question: {avg_num_options}\n")

    # Stats of videos corresponding to the questions
    unique_videos = set()
    for annotation in annotations:
        unique_videos.add(annotation["vid_filename"])
    print(f"Number of unique videos corresponding to the questions: {len(unique_videos)}")
    # Using FFMPEG, get the duration of each video, and find the longest, shortest and average duration
    video_durations = []
    for video_filename in unique_videos:
        video_path = f"./dataset/SUTD-TrafficQA/compressed_videos/{video_filename}"
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
        video_durations.append(duration)
    print(f"Longest video duration: {max(video_durations)} seconds")
    print(f"Shortest video duration: {min(video_durations)} seconds")
    avg_duration = sum(video_durations) / len(video_durations)
    print(f"Average video duration: {avg_duration} seconds")

    # Print sorted list of durations
    #video_durations.sort()
    #print("Sorted list of video durations:")
    #for duration in video_durations:
    #    print(f"{duration} seconds")


# Describe the dataset if the script is executed directly
if __name__ == "__main__":
    describe_sutd_traffic_qa_dataset_subset(NUM_SAMPLES)
