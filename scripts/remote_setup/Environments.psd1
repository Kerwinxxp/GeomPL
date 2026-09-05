@{
    GeoRanker = @{ Requirements='requirements-georanker.txt'; Freeze='georanker.freeze.txt'; Imports='belief_elicit.georanker_belief,belief_elicit.run_georanker_inpaint,belief_elicit.distributed_georanker' }
    Sam3 = @{ Requirements='requirements-sam3.txt'; Freeze='sam3.freeze.txt'; Imports='transformers.models.sam3.modeling_sam3,cue_extract.extract_vocab' }
    LaMa = @{ Requirements='requirements-lama.txt'; Freeze='lama.freeze.txt'; Imports='simple_lama_inpainting,belief_elicit.precompute_inpaint' }
}
