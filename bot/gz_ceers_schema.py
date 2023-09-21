'''
Yuanzhe Dong, 21 Sept at IAC
'''


from zoobot.shared.schemas import Schema



gz_ceers_pairs = {
    't0_smooth_or_featured': ['__features_or_disk', '__smooth', '__star_artifact_or_bad_zoom'],
    't1_how_rounded_is_it': ['__cigarshaped', '__in_between', '__completely_round'],
    't2_could_this_be_a_disk_viewed_edgeon': ['__yes_edge_on_disk', '__no_something_else'],
    't3_edge_on_bulge_what_shape': ['__boxy', '__no_bulge', '__rounded'],
    't4_is_there_a_bar': ['__no_bar', '__strong_bar', '__weak_bar'],
    't5_is_there_any_spiral_arm_pattern': ['__yes', '__no'],
    't6_spiral_how_tightly_wound': ['__loose', '__medium', '__tight'],
    't7_how_many_spiral_arms_are_there': ['__1', '__2', '__3', '__4', '__more_than_4', '__cant_tell'],
    't8_not_edge_on_bulge': ['__dominant', '__moderate', '__no_bulge', '__large', '__small'],
    't11_is_the_galaxy_merging_or_disturbed': ['__major_disturbance', '__merging', '__minor_disturbance', '__none'],
    't12_are_there_any_obvious_bright_clumps': ['__yes', '__no'],
    't19_what_problem_do___e_with_the_image': ['__nonstar_artifact', '__bad_image_zoom', '__star'],
}

gz_ceers_dependencies = {
    't0_smooth_or_featured': None,
    't1_how_rounded_is_it': 't0_smooth_or_featured__smooth',
    't2_could_this_be_a_disk_viewed_edgeon': 't0_smooth_or_featured__features_or_disk',
    't3_edge_on_bulge_what_shape': 't2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk',
    't4_is_there_a_bar': 't2_could_this_be_a_disk_viewed_edgeon__no_something_else',
    't5_is_there_any_spiral_arm_pattern': 't2_could_this_be_a_disk_viewed_edgeon__no_something_else',
    't6_spiral_how_tightly_wound': 't5_is_there_any_spiral_arm_pattern__yes',
    't7_how_many_spiral_arms_are_there': 't5_is_there_any_spiral_arm_pattern__yes',
    't8_not_edge_on_bulge': 't2_could_this_be_a_disk_viewed_edgeon__no_something_else',
    't11_is_the_galaxy_merging_or_disturbed': None,     # double dependency not supported
    't12_are_there_any_obvious_bright_clumps': 't0_smooth_or_featured__features_or_disk',
    't19_what_problem_do___e_with_the_image': 't0_smooth_or_featured__star_artifact_or_bad_zoom'
}

gz_ceers_schema = Schema(gz_ceers_pairs,gz_ceers_dependencies)