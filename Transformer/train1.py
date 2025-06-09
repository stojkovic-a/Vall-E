import torch
from transformer import NARVALLE, ARVALLE
from dataloader import get_dataloader
import config as conf
from tqdm import tqdm
import torch.nn as nn
import os
from pathlib import Path
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,garbage_collection_threshold:0.1,max_split_size_mb:21"
)
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


def save_models(ar, nar, opt, schedular, num_updates, permanent=True):
    if permanent == True:
        num_folders = len(os.listdir(conf.MODEL_SAVE_DIR))
        save_path = Path(conf.MODEL_SAVE_DIR) / str(num_folders + 1)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ar": ar.state_dict(),
                "nar": nar.state_dict(),
                "opt": opt.state_dict(),
                "schedular": schedular.state_dict(),
            },
            save_path / f"{num_updates}.pt",
        )
    else:
        torch.save(
            {
                "ar": ar.state_dict(),
                "nar": nar.state_dict(),
                "opt": opt.state_dict(),
                "schedular": schedular.state_dict(),
            },
            f"temp.pt",
        )


if __name__ == "__main__":
    device = "cuda"
    dataloader, dataset = get_dataloader(
        conf.PHONEME_DIR,
        conf.QNTS_DIR,
        conf.PHONEME_SUFFIX,
        conf.QNTS_SUFFIX,
        conf.BATCH_SIZE,
    )
    phonemes_vocab_size = dataset._get_phonem_vocab_size()
    qnts_vocab_size = dataset._get_qnt_vocab_size()
    ar = ARVALLE(
        phonemes_vocab_size,
        qnts_vocab_size,
        conf.EMBEDDING_DIMENSION,
        conf.NUM_HEADS,
        conf.NUM_LAYERS,
        conf.FF,
        conf.MAX_SEQ_LENGTH,
        conf.DROPOUT,
    ).to(device)
    nar = NARVALLE(
        phonemes_vocab_size,
        qnts_vocab_size,
        8,
        conf.EMBEDDING_DIMENSION,
        conf.NUM_HEADS,
        conf.NUM_LAYERS,
        conf.FF,
        conf.MAX_SEQ_LENGTH,
        conf.DROPOUT,
    ).to(device)
    opt = torch.optim.SGD(list(ar.parameters()) + list(nar.parameters()), conf.LR)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.3, patience=6000
    )
    load = True
    if load == True:
        d = torch.load("./temp.pt")
        ar.load_state_dict(d["ar"])
        nar.load_state_dict(d["nar"])
        # opt.load_state_dict(d["opt"])
        # schedular.load_state_dict(d["schedular"])
        del d
        torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    ar.train()
    nar.train()
    num_updates = 0
    pad_p = torch.full(
        (conf.BATCH_SIZE, 1), phonemes_vocab_size - 1, dtype=torch.int64
    ).to(device)
    pad_q = torch.full((conf.BATCH_SIZE, 8, 1), qnts_vocab_size - 1).to(device)
    # stage_tensors = [torch.tensor(i, device=device) for i in range(1, 8)]
    for epoch in range(conf.NUM_EPOCHES):
        loop = tqdm(dataloader, leave=False)
        for batch_idx, (p, q) in enumerate(loop):
            # print(
            #     torch.cuda.memory_allocated()
            #     / (torch.cuda.max_memory_allocated() + 0.0000001)
            # )
            opt.zero_grad()
            with torch.no_grad():
                p = torch.tensor(p, dtype=torch.int64).to(device)
                p = torch.cat([p, pad_p], dim=1)
                # p = torch.cat(
                #     [
                #         p,
                #         torch.full((p.shape[0], 1), phonemes_vocab_size - 1).to(device),
                #     ],
                #     dim=1,
                # )
                q = torch.stack(q).to(device)
                q = torch.cat([q, pad_q], dim=2)

                print(p.shape)
                print(q.shape)
                print("\n")

                # q = torch.cat(
                #     [
                #         q,
                #         torch.full((q.shape[0], q.shape[1], 1), qnts_vocab_size - 1).to(
                #             device
                #         ),
                #     ],
                #     dim=2,
                # )

                if q.shape[2] <= 301 or q.shape[2]>1248:
                    continue
                # 10 seconds q are 750, threfore for 4 seconds 300
            ar_q_predicted = ar(p, q[:, 0, :].squeeze(1), 300)
            ar_q_predicted_useful = torch.transpose(ar_q_predicted, 1, 2).contiguous()[
                :, :, p.shape[1] + 299 : -1
            ]
            ar_q_ground_truth = q[:, 0, 300:].squeeze(1)
            ar_loss = criterion(ar_q_predicted_useful, ar_q_ground_truth)

            stage = (batch_idx % 7) + 1
            stage = torch.tensor(stage).to(device)
            # stage = stage_tensors[batch_idx % 7]
            nar_q_predicted = nar(p, q[:, :, :300], q[:, :, 300:], stage)
            nar_q_predicted_useful = torch.transpose(
                nar_q_predicted, 1, 2
            ).contiguous()[:, :, p.shape[1] + 300 :]
            nar_q_ground_truth = q[:, stage, 300:].squeeze(1)
            nar_loss = criterion(nar_q_predicted_useful, nar_q_ground_truth)

            loss = ar_loss + nar_loss
            loss.backward()
            opt.step()
            l = loss.detach().to("cpu").numpy()
            schedular.step(l)
            with torch.no_grad():
                num_updates += 1
                with open("losses.txt", "a") as f:
                    f.write(f"{l} ")
                if num_updates % conf.UPDATES_PER_SAVE == 0:
                    save_models(ar, nar, opt, schedular, num_updates, permanent=True)
                elif num_updates % conf.TEMP_SAVE == 0:
                    save_models(ar, nar, opt, schedular, num_updates, permanent=False)

            del (
                stage,
                ar_q_predicted_useful,
                ar_q_ground_truth,
                ar_q_predicted,
                nar_q_predicted_useful,
                nar_q_ground_truth,
                nar_q_predicted,
                p,
                q,
                ar_loss,
                nar_loss,
                loss,
                l,
            )
            torch.cuda.empty_cache()
            gc.collect()
            # # print(gc.get_objects())
            # # print(torch.cuda.memory_summary())
