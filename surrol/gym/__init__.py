from gym.envs.registration import register


# PSM Env
register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieve-v0',
    entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=50,
)

register(
    id='PegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer:PegTransfer',
    max_episode_steps=50,
)

# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    max_episode_steps=50,
)

# ECM Env
register(
    id='ECMReach-v0',
    entry_point='surrol.tasks.ecm_reach:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    entry_point='surrol.tasks.ecm_misorient:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    max_episode_steps=500,
)

register(
    id='GauzeRetrieveMem-v0',  # Ensure it follows Gym's naming convention
    entry_point='surrol.tasks.gauze_retrieve_mem:GauzeRetrieveMem',  # Path to your custom environment
    max_episode_steps=50,
)

register(
    id='NeedlePickMem-v0',
    entry_point='surrol.tasks.needle_pick_mem:NeedlePickMem',
    max_episode_steps=50,
)

register(
    id='NeedleReachMem-v0',
    entry_point='surrol.tasks.needle_reach_mem:NeedleReachMem',
    max_episode_steps=50,
)

register(
    id='NeedleRegraspMem-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual_mem:NeedleRegraspMem',
    max_episode_steps=50,
)

register(
    id='BiPegTransferMem-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual_mem:BiPegTransferMem',
    max_episode_steps=50,
)

register(
    id='PegTransferMem-v0',
    entry_point='surrol.tasks.peg_transfer_mem:PegTransferMem',
    max_episode_steps=50,
)

register(
    id='MisOrientMem-v0',
    entry_point='surrol.tasks.ecm_misorient_mem:MisOrientMem',
    max_episode_steps=50,
)
register(
    id='ECMReachMem-v0',
    entry_point='surrol.tasks.ecm_reach_mem:ECMReachMem',
    max_episode_steps=50,
)
register(
    id='StaticTrackMem-v0',
    entry_point='surrol.tasks.ecm_static_track_mem:StaticTrackMem',
    max_episode_steps=50,
)