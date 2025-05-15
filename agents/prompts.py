from string import Template

SPEAKER_SYSTEM_PROMPT_BASIC = Template(
    "I will show  $num_images images labelled as $labels. I will then mention an image. Describe the image corresponding to the label. Your response should only contain the message. Your message does not need to be a full sentence. Your message should be a brief description of something in the image."
)

SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC = (
    "Generate a message referring to one of the images."
)

SPEAKER_USER_PROMPT_TARGET_BASIC = Template(
    "Describe Image $target. Generate only a message containing $content. Do not include the label of the image in your message."
)

SPEAKER_SYSTEM_PROMPT_STANDARD = Template(
    "Play a repeated reference game with me and a third player (the listener). You will act as the speaker in the game. This game consists of multiple rounds in which the speaker interacts with me and the listener on the same referential context ($num_images images). In each round, I give you one of the $num_images images as the target. You should communicate the target to the listener in a message. The listener will try to choose the target correctly based on your message. I will tell you which image the listener chooses. The listener will see the $num_images images in a different order every round so you cannot communicate the target simply by using its position or label ($labels).\n\nYour reply should only contain the message. Your message does not need to be a full sentence."
)

SPEAKER_SYSTEM_PROMPT_EXPLICIT = Template(
    "Play a repeated reference game with me and a third player (the listener). You will act as the speaker in the game. This game consists of multiple rounds in which the speaker interacts with me and the listener on the same referential context ($num_images images). In each round, I give you one of the $num_images images as the target. You should communicate the target to the listener in a message. The listener will try to choose the target correctly based on your message. I will tell you which image the listener chooses. The listener will see the $num_images images in a different order every round so you cannot communicate the target simply by using its position or label ($labels).\n\nYour reply should only contain the message and the message should always be shorter than 20 words. Throughout, your message should not exceed one sentence but it does not need to be a full sentence. Start with more detailed messages to ensure the listener's accuracy. As more rounds are completed and the listener understands you better, gradually condense your messages, making them shorter and shorter every round. When creating a shorter message for an image, try to extract salient tokens from the previous messages for this image rather than introducing new words. The short messages should still allow the listener to choose the target correctly. For each image, when you reach a message you think can not be further shortened without hurting the listener's accuracy, you should keep using that message for the rest of the game."
)

LISTENER_SYSTEM_PROMPT = "You are an assistant who will play a series of reference games with the user. You will pay close attention to the conversation history as more rounds are played."

LISTENER_USER_PROMPT = Template(
    "Play a game with multiple rounds involving the same set of images. In each round, I will refer to one of the images with a message. You will guess which image I am referring to. If present, the history of previous rounds may help you better understand how I refer to specific images. In each round, answer with the image's label, i.e. one of [$labels]. You should still make a guess even when you are not sure. Do not output anything other than the image label you guess."
)

SPEAKER_USER_PROMPT_PHOTOGRAPHS = "You are an assistant who will play a series of reference games with the user. You will generate a message referring to one of the images. The user will guess which image you are referring to."

SPEAKER_USER_PROMPT_TANGRAMS = "You are an assistant who will play a series of reference games with the user. You will generate a message referring to one of the images. The user will guess which image you are referring to. The images are of tangram shapes. Try to avoid referring to specific pieces of the tangram. Try to describe the shape as a whole. Feel free to use the resemblance to any real-world objects, and parts of those real world objects to describe the image."

SPEAKER_USER_PROMPT_TARGET = Template(
    "The target image is $target. Describe the target image to the listener. Generate only a message containing $content."
)
