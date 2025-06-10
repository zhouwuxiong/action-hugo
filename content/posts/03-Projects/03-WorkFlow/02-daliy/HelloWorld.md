---
date: 2024-02-02T04:14:54-08:00
title: HelloWorld
share: true
---


# 流程 124141
1. obsidian 安装  `Enveloppe`

## 1 hugo (静态网站模板生称工具)
```sh
sudo snap install hugo
# disable automatic updates
sudo snap refresh --hold hugo

hugo new site quickstart 

hugo version

```
## 2 obsidian 安装  Enveloppe


## 3 Enveloppe 配置
复制 json 文件导入
[obsidian-github-publisher-hugo/settings.json at main · miaogaolin/obsidian-github-publisher-hugo · GitHub](https://github.com/miaogaolin/obsidian-github-publisher-hugo/blob/main/settings.json)
### 3.1 Enveloppe 配置github
![[2-obsidian blog publish.png|2-obsidian blog publish.png]]

1. fork 网站模板
2. 获取 github token
![[1-obsidian blog publish.png|1-obsidian blog publish.png]]

### 3.2 目录配置 
![[3-obsidian blog publish.png|3-obsidian blog publish.png]]

## 4 vecel 部署
![[4-obsidian blog publish.png|4-obsidian blog publish.png]]


default config
```json
{
  "github": {
    "branch": "main",
    "automaticallyMergePR": true,
    "dryRun": {
      "enable": false,
      "folderName": "enveloppe"
    },
    "api": {
      "tiersForApi": "Github Free/Pro/Team (default)",
      "hostname": ""
    },
    "workflow": {
      "commitMessage": "[PUBLISHER] Merge",
      "name": ""
    },
    "verifiedRepo": true
  },
  "upload": {
    "behavior": "fixed",
    "defaultName": "content/posts",
    "rootFolder": "",
    "yamlFolderKey": "",
    "frontmatterTitle": {
      "enable": true,
      "key": "title"
    },
    "replaceTitle": [],
    "replacePath": [],
    "autoclean": {
      "includeAttachments": true,
      "enable": false,
      "excluded": []
    },
    "folderNote": {
      "enable": false,
      "rename": "index.md",
      "addTitle": {
        "enable": false,
        "key": "title"
      }
    },
    "metadataExtractorPath": ""
  },
  "conversion": {
    "hardbreak": false,
    "dataview": true,
    "censorText": [
      {
        "entry": "/\\]\\(([^/\\)]+?)\\.(png|jpg|jpeg|webp|gif|avif)/",
        "replace": "](/images/$1.$2",
        "flags": "",
        "after": true
      }
    ],
    "tags": {
      "inline": false,
      "exclude": [],
      "fields": []
    },
    "links": {
      "internal": false,
      "unshared": false,
      "wiki": false,
      "slugify": "disable",
      "unlink": false,
      "relativePath": true,
      "textPrefix": "/"
    }
  },
  "embed": {
    "attachments": true,
    "overrideAttachments": [],
    "keySendFile": [],
    "notes": false,
    "folder": "",
    "convertEmbedToLinks": "keep",
    "charConvert": "->",
    "unHandledObsidianExt": [],
    "sendSimpleLinks": true,
    "forcePush": true
  },
  "plugin": {
    "shareKey": "share",
    "excludedFolder": [],
    "copyLink": {
      "enable": false,
      "links": "",
      "removePart": [],
      "addCmd": false,
      "transform": {
        "toUri": true,
        "slugify": "lower",
        "applyRegex": []
      }
    },
    "setFrontmatterKey": "Set"
  }
}
```

text replace 用于处理文件路径格式 [Enveloppe 插件适配 Hugo 的配置 | PrintLove](https://www.printlove.cn/github-publisher-hugo/)
```json
    "censorText": [
      {
        "entry": "/\\]\\(([^)\\.]+)\\.md/",
        "replace": "]({{< relref \"$1.md\" >}}",
        "flags": "",
        "after": true,
        "inCodeBlocks": false
      },
      {
        "entry": "/cover\\.image/",
        "replace": "cover:\\n    image",
        "flags": "",
        "after": false
      },
      {
        "entry": "/\\]\\(([^/\\)]+?)\\.(png|jpg|jpeg|webp|gif)/",
        "replace": "](/images/$1.$2",
        "flags": "",
        "after": true
      },
      {
        "entry": "/\\!\\[\\[([^/\\]]+?)\\.(png|jpg|jpeg|webp|gif)\\|(\\d+)(x(\\d+))?\\]\\]/",
        "replace": "{{< figure src=\"/images/$1.$2\"  width=\"$3\" height=\"$5\">}}",
        "flags": "",
        "after": false
      },
      {
        "entry": "/\\!\\[\\[([^/\\]]+?)\\.(png|jpg|jpeg|webp|gif)\\|([^\\|]*?)(\\|(\\d+)(x(\\d+))?)?\\]\\]/",
        "replace": "{{< figure src=\"/images/$1.$2\" caption=\"$3\" width=\"$5\" height=\"$7\">}}",
        "flags": "",
        "after": false
      }
    ],
```


final config
```json
{
  "github": {
    "branch": "main",
    "automaticallyMergePR": true,
    "dryRun": {
      "enable": false,
      "folderName": "enveloppe"
    },
    "api": {
      "tiersForApi": "Github Free/Pro/Team (default)",
      "hostname": ""
    },
    "workflow": {
      "commitMessage": "[PUBLISHER] Merge",
      "name": ""
    },
    "verifiedRepo": true
  },
  "upload": {
    "behavior": "fixed",
    "defaultName": "content/posts",
    "rootFolder": "",
    "yamlFolderKey": "",
    "frontmatterTitle": {
      "enable": true,
      "key": "title"
    },
    "replaceTitle": [],
    "replacePath": [],
    "autoclean": {
      "includeAttachments": true,
      "enable": false,
      "excluded": []
    },
    "folderNote": {
      "enable": false,
      "rename": "index.md",
      "addTitle": {
        "enable": false,
        "key": "title"
      }
    },
    "metadataExtractorPath": ""
  },
  "conversion": {
    "hardbreak": false,
    "dataview": true,
    "censorText": [
      {
        "entry": "/\\]\\(([^)\\.]+)\\.md/",
        "replace": "]({{< relref \"$1.md\" >}}",
        "flags": "",
        "after": true,
        "inCodeBlocks": false
      },
      {
        "entry": "/cover\\.image/",
        "replace": "cover:\\n    image",
        "flags": "",
        "after": false
      },
      {
        "entry": "/\\]\\(([^/\\)]+?)\\.(png|jpg|jpeg|webp|gif)/",
        "replace": "](../../images/$1.$2",
        "flags": "",
        "after": true
      },
      {
        "entry": "/\\!\\[\\[([^/\\]]+?)\\.(png|jpg|jpeg|webp|gif)\\|(\\d+)(x(\\d+))?\\]\\]/",
        "replace": "{{< figure src=\"/images/$1.$2\"  width=\"$3\" height=\"$5\">}}",
        "flags": "",
        "after": false
      },
      {
        "entry": "/\\!\\[\\[([^/\\]]+?)\\.(png|jpg|jpeg|webp|gif)\\|([^\\|]*?)(\\|(\\d+)(x(\\d+))?)?\\]\\]/",
        "replace": "{{< figure src=\"/images/$1.$2\" caption=\"$3\" width=\"$5\" height=\"$7\">}}",
        "flags": "",
        "after": false
      }
    ],
    "tags": {
      "inline": false,
      "exclude": [],
      "fields": []
    },
    "links": {
      "internal": false,
      "unshared": false,
      "wiki": false,
      "slugify": "disable",
      "unlink": false,
      "relativePath": true,
      "textPrefix": "/"
    }
  },
  "embed": {
    "attachments": true,
    "overrideAttachments": [],
    "keySendFile": [],
    "notes": false,
    "folder": "",
    "convertEmbedToLinks": "keep",
    "charConvert": "->",
    "unHandledObsidianExt": [],
    "sendSimpleLinks": true,
    "forcePush": true
  },
  "plugin": {
    "shareKey": "share",
    "excludedFolder": [],
    "copyLink": {
      "enable": false,
      "links": "",
      "removePart": [],
      "addCmd": false,
      "transform": {
        "toUri": true,
        "slugify": "lower",
        "applyRegex": []
      }
    },
    "setFrontmatterKey": "Set"
  }
}
```

```
{
  "github": {
    "branch": "main",
    "automaticallyMergePR": true,
    "dryRun": {
      "enable": false,
      "folderName": "enveloppe"
    },
    "api": {
      "tiersForApi": "Github Free/Pro/Team (default)",
      "hostname": ""
    },
    "workflow": {
      "commitMessage": "[PUBLISHER] Merge",
      "name": ""
    },
    "verifiedRepo": true
  },
  "upload": {
    "behavior": "obsidian",
    "defaultName": "content/posts",
    "rootFolder": "../",
    "yamlFolderKey": "",
    "frontmatterTitle": {
      "enable": true,
      "key": "title"
    },
    "replaceTitle": [],
    "replacePath": [],
    "autoclean": {
      "includeAttachments": true,
      "enable": false,
      "excluded": []
    },
    "folderNote": {
      "enable": false,
      "rename": "index.md",
      "addTitle": {
        "enable": false,
        "key": "title"
      }
    },
    "metadataExtractorPath": ""
  },
  "conversion": {
    "hardbreak": false,
    "dataview": true,
    "censorText": [
      {
        "entry": "/\\]\\(([^)\\.]+)\\.md/",
        "replace": "]({{< relref \"$1.md\" >}}",
        "flags": "",
        "after": true,
        "inCodeBlocks": false
      },
      {
        "entry": "/cover\\.image/",
        "replace": "cover:\\n    image",
        "flags": "",
        "after": false
      },
      {
        "entry": "/\\]\\(([^/\\)]+?)\\.(png|jpg|jpeg|webp|gif)/",
        "replace": "](../../images/$1.$2",
        "flags": "",
        "after": true
      },
      {
        "entry": "/\\!\\[\\[([^/\\]]+?)\\.(png|jpg|jpeg|webp|gif)\\|(\\d+)(x(\\d+))?\\]\\]/",
        "replace": "{{< figure src=\"/images/$1.$2\"  width=\"$3\" height=\"$5\">}}",
        "flags": "",
        "after": false
      },
      {
        "entry": "/\\!\\[\\[([^/\\]]+?)\\.(png|jpg|jpeg|webp|gif)\\|([^\\|]*?)(\\|(\\d+)(x(\\d+))?)?\\]\\]/",
        "replace": "{{< figure src=\"/images/$1.$2\" caption=\"$3\" width=\"$5\" height=\"$7\">}}",
        "flags": "",
        "after": false
      }
    ],
    "tags": {
      "inline": false,
      "exclude": [],
      "fields": []
    },
    "links": {
      "internal": false,
      "unshared": false,
      "wiki": false,
      "slugify": "disable",
      "unlink": false,
      "relativePath": true,
      "textPrefix": "/"
    }
  },
  "embed": {
    "attachments": true,
    "overrideAttachments": [],
    "keySendFile": [],
    "notes": false,
    "folder": "",
    "convertEmbedToLinks": "keep",
    "charConvert": "->",
    "unHandledObsidianExt": [],
    "sendSimpleLinks": true,
    "forcePush": true
  },
  "plugin": {
    "shareKey": "share",
    "excludedFolder": [],
    "copyLink": {
      "enable": false,
      "links": "",
      "removePart": [],
      "addCmd": false,
      "transform": {
        "toUri": true,
        "slugify": "lower",
        "applyRegex": []
      }
    },
    "setFrontmatterKey": "Set"
  }
}
```