from lets_try import detect_gender

# video_name = 'vid1015'    # cooking
# video_name = 'vid153'     # man and women baaad
# video_name = 'vid217'     # no person
# video_name = 'vid283'     # two males
# video_name = 'vid1934'    # sax
# video_name = 'vid1956'
# video_name = 'vid238'     # three girls baaad
# video_name = 'vid264'     # one girl
# video_name = 'vid284'     # two man cooking
# video_name = 'vid774'     # one woman
# video_name = 'vid787'     # two women
# video_name = 'vid867'     # two women
# video_name = 'vid899'     # one woman one man, recognizes two man bad quality, two scenes
video_name = 'vid921'     # two men one scene and one man other scene
# video_name = 'vid222'     # baby
# video_name = 'vid240'     # baby baaad
# video_name = 'vid257'     # child baad
# video_name = 'vid271'     # child
# video_name = 'vid691'     # one woman
# video_name = 'vid812'     # child f
# video_name = 'vid876'     # baby
# video_name = 'vid1212'    # three child m
# video_name = 'vid1298'    # child m
# video_name = 'vid130'     # old man
# video_name = 'vid1410'    # two woman cooking
# video_name = 'vid1564'    # baby and women
# video_name = 'vid1604'    # one man
# video_name = 'vid1628'    # one woman
# video_name = 'vid1630'    # one woman
# video_name = 'vid1635'    # a lot of people
# video_name = 'vid1683'    # one baby
# video_name = 'vid1726'    # two men
# video_name = 'vid1749'    # old f, adult m
# video_name = 'vid1763'    # old f, adult m
# video_name = 'vid1862'    # child f
# video_name = 'vid1865'    # baby m
# video_name = 'vid1937'    # various child/baby
# video_name = 'vid1967'    # child f, child/baby m
# video_name = 'vid1953'

# FRAMES = '/home/users/ramosrafh/git/YoloKerasFaceDetection/frames/'
FRAMES = '/home/users/datasets/youtubeclips-datasetV2/frames/' + video_name + '/'

people = detect_gender(FRAMES)

print(people)
