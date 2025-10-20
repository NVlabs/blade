## BEDLAM-CC Dataset Generation Guide (Windows Only)

all files mentioned are contained in `docs/bedlamcc_assets.zip`

### Setup
- Follow the Unreal rendering setup for the original BEDLAM.
- Copy `*.py` files from our repo to `.\Content\PS\Bedlam\Core\Python`.
- Copy updated `BEDLAMCC.uasset` render control to `/All/EngineData/Engine/PS/Core/EditorScripting/`.
- Copy render config `NvBedlamConfig.uasset` to the Unreal BEDLAM project at `/All/Game`.
- Copy `SMPLX_NEUTRAL.npz` from the SMPL-X project page to the dataset folder `C:\blade\assets\smplx`.
- Copy BEDLAM animation data (`*.npz`) to the dataset folder (default: `C:\blade\animations\gendered_ground_truth\sequencename\moving_body_para\%04d\motion_seq.npz`).
- Configure Unreal Python environment:

```bash
pip install numpy smplx[all]
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Generation Process
1. Open Unreal BEDLAM project.
2. Open BEDLAM render control at `/All/EngineData/Engine/PS/Core/EditorScripting/BEDLAMCC`.
3. Right-click → run `Editor Utility Widget` (see first screenshot).
    <p align="center">
      <img src="docs/bedlamcc_1_editorutilitywidget.png" alt="BLADE pipeline" width="70%">
    </p>

#### Generate level sequences
1. Delete previously existing level sequences in asset folder `/All/Content/Bedlam/LevelSequences`.
2. Add target folder path holding configuration CSV files to the first text line. Default: `C:\blade`.
3. Specify CSV file to generate sequences from (one of the provided for BLADE/BEDLAM-CC):
   - `blade_seq_1_1.csv`
   - `blade_seq_1_10.csv`
   - `blade_seq_1_10000.csv`
   - `blade_seq_1_100000.csv`
4. Specify render preset (e.g., `NvBedlamConfig`).
5. Click `Create LevelSequences` (see second screenshot).
    <p align="center">
      <img src="docs/bedlamcc_2_create_levelsequences.png" alt="BLADE pipeline" width="70%">
    </p>
   
   - This process can take time for a large number of sequences.
   - Check the output log for any error messages.
   - The button should turn green if successful.
6. Check if output folders have been generated:
   - `/dataset_folder/info` → holding metadata (camera poses, animation data).

#### Render level sequences
1. Select target level sequences (or `CTRL+A` to select all) in the content browser under `/All/Game/Bedlam/LevelSequences`.
2. Press `CreateMovieRenderQueue` to fill the Movie Render Queue.
3. Check for the correct output folder in the Movie Render Queue GUI.
4. Press `Render (ground truth export)`.
5. Output folders are generated (see third screenshot):
    <p align="center">
      <img src="docs/bedlamcc_3_render.png" alt="BLADE pipeline" width="70%">
    </p>
   
   - `./images/seq_*` → holding rendered images (RGB + normal).