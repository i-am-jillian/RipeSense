"""Download and verify the pinned local image/text model (about 149 MiB)."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION="6b7c523703ae26e39be4b03774966f8571c7f523"
FILES={
    "onnx/text_model_quantized.onnx":"73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
    "onnx/vision_model_quantized.onnx":"583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
    "tokenizer.json":"f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
    "tokenizer_config.json":"60ba2912bc6344c94bc16bbdec27fa1209409167b6f2fdf3cfe9e65462ea3967",
    "preprocessor_config.json":"6f638fb9401a6d6296feff533ee7efe657b787c49f954f82f5906b36ef2a1b1f",
    "config.json":"493ef57ff783e42d1530c91b53469b7fdf8db8a9c1408e86998fcb7899a4f495",
    "README.md":"ae287015b60eb7ca1d40e726c550af1988977c24f0005f123274d52490c20d8d",
}


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[2]/"models/openvision"
    root.mkdir(parents=True,exist_ok=True)
    def download(pair):
        name,sha=pair
        dest=root/Path(name).name
        url=f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/{REVISION}/{name}"
        if not dest.exists() or digest(dest)!=sha:
            temporary=dest.with_suffix(dest.suffix+".partial")
            with urllib.request.urlopen(url,timeout=120) as response, temporary.open("wb") as stream:
                for block in iter(lambda:response.read(1024*1024),b""):
                    stream.write(block)
            if digest(temporary)!=sha:
                raise ValueError(f"Model checksum failed: {name}")
            temporary.replace(dest)
        print("Verified",dest.name,flush=True)
        return {"file":dest.name,"sha256":sha,"url":url}
    with ThreadPoolExecutor(max_workers=3) as executor:
        files=list(executor.map(download,FILES.items()))
    (root/"source.json").write_text(json.dumps({"repository":"Xenova/clip-vit-base-patch32",
                                               "revision":REVISION,"files":files},indent=2))


if __name__=="__main__":
    main()
