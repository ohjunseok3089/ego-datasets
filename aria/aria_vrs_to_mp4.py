from projectaria_tools.utils.vrs_to_mp4_utils import convert_vrs_to_mp4
import os

seqs = [
    "loc1_script2_seq1_rec1", "loc1_script2_seq1_rec2", "loc1_script2_seq3_rec1",
    "loc1_script2_seq3_rec2", "loc1_script2_seq4_rec1", "loc1_script2_seq4_rec2",
    "loc1_script2_seq6_rec1", "loc1_script2_seq6_rec2", "loc1_script2_seq7_rec1",
    "loc1_script2_seq7_rec2", "loc1_script2_seq8_rec1", "loc1_script2_seq8_rec2",
    "loc1_script3_seq2_rec1", "loc2_script2_seq1_rec1", "loc2_script2_seq1_rec2",
    "loc2_script2_seq3_rec1", "loc2_script2_seq3_rec2", "loc2_script2_seq4_rec1",
    "loc2_script2_seq5_rec1", "loc2_script2_seq5_rec2", "loc2_script2_seq6_rec1",
    "loc2_script2_seq6_rec2", "loc2_script2_seq8_rec1", "loc2_script2_seq8_rec2",
    "loc2_script3_seq1_rec2", "loc2_script3_seq2_rec1", "loc2_script3_seq2_rec2",
    "loc2_script3_seq4_rec1", "loc2_script3_seq4_rec2", "loc3_script2_seq1_rec1",
    "loc3_script2_seq1_rec2", "loc3_script2_seq3_rec1", "loc3_script2_seq3_rec2",
    "loc3_script2_seq4_rec1", "loc3_script2_seq4_rec2", "loc3_script2_seq5_rec1",
    "loc3_script2_seq5_rec2", "loc3_script2_seq7_rec1", "loc3_script2_seq7_rec2",
    "loc3_script3_seq1_rec2", "loc3_script3_seq2_rec1", "loc3_script3_seq2_rec2",
    "loc3_script3_seq4_rec1", "loc3_script3_seq4_rec2"
]

for seq in seqs:
    vrs_path = os.path.join("/mas/robots/prg-aria/raw", seq, "recording.vrs")
    mp4_path = os.path.join("/mas/robots/prg-aria/dataset", f"{seq}.mp4")
    if os.path.isfile(vrs_path):
        print(f"Converting {vrs_path} to {mp4_path} ...")
        convert_vrs_to_mp4(vrs_path, mp4_path)
    else:
        print(f"VRS file not found: {vrs_path}")
