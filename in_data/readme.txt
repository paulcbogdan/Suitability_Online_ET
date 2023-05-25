calculate_ET.py converts the "Study1/2_online_data_anonimized.csv" files into "Study1/2_online_data_anonimized_gaze.csv" files.

The former files contain the following columns:
	subject = integer; code indicating participant number
	excluded = boolean; 1 if participant was excluded following published procedures; 0 otherwise
	img = string; name of image (.jpg)
	EmoRating = integer; ranges 1-5, rating of arousal in response to the image
	emotionality = string; 'emo' or 'neu'
	attn = string; 'FV' (free-viewing; Study 1), 'FG' (foreground; Study 2) or 'BG' (background; Study 3)
	study = string; 'study1' or 'study2'
	bad_trial = boolean; 1 if FG_time + BG_time < .1, i.e., less than 10% of gaze points are within the image (foreground or background); 0 otherwise
	IA_name = string; name of interest area, which is simply the image name with a .ias extension

The converted files also contain:
	FG_time = float; 0-1; proportion of gaze points recorded within the foreground
	BG_time = float; 0-1; proportion of gaze points recorded within the background; note, FG_time + BG_time will not equal 1.0 if some gaze points fall outside the image
	FG_propo = float; FG_time / (FG_time + BG_time)

