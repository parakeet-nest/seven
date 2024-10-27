package main

/*
https://www.anthropic.com/news/contextual-retrieval
*/

import (
	"fmt"
	"log"
	"strconv"

	"github.com/parakeet-nest/parakeet/cli"
	"github.com/parakeet-nest/parakeet/completion"
	"github.com/parakeet-nest/parakeet/content"
	"github.com/parakeet-nest/parakeet/embeddings"
	"github.com/parakeet-nest/parakeet/enums/option"
	"github.com/parakeet-nest/parakeet/llm"
	"github.com/parakeet-nest/parakeet/ui"
	"github.com/parakeet-nest/parakeet/ui/colors"

	"github.com/sea-monkeys/daphnia"
)

func createEmbeddings(docsPath, storePath, ollamaUrl, embeddingsModel, contextualModel string) {

	store := daphnia.VectorStore{}
	err := store.Initialize(storePath)

	//store := embeddings.BboltVectorStore{}
	//err := store.Initialize(storePath)
	if err != nil {
		log.Fatalln("😡:", err)
	}

	content.ForEachFile(docsPath, ".md", func(documentPath string) error {
		fmt.Println("📝 Creating embedding from document ", documentPath)

		// Read the content of the file
		wholeDocumentContent, err := content.ReadTextFile(documentPath)
		if err != nil {
			log.Fatalln("😡:", err)
		}

		chunks := content.ParseMarkdownWithLineage(wholeDocumentContent)

		fmt.Println("👋 Found", len(chunks), "chunks")

		// Create embeddings from documents and save them in the store
		for idx, doc := range chunks {
			fmt.Println("📝 Creating embedding from document ", idx)
			fmt.Println("🖼️", doc.Header)

			// better chunk embedding
			options := llm.SetOptions(map[string]interface{}{
				option.Temperature: 0.8,
			})

			context, err := content.CreateChunkContext(wholeDocumentContent, doc, ollamaUrl, contextualModel, options)
			if err != nil {
				log.Println("😡:", err)
			}
			ui.Println(colors.Green, "---[Chunk context]--------------------------------")
			ui.Println(colors.Green, context)
			ui.Println(colors.Green, "--------------------------------------------------")

			embedding, err := embeddings.CreateEmbedding(
				ollamaUrl,
				llm.Query4Embedding{
					Model:  embeddingsModel,
					Prompt: context + "\n" + fmt.Sprintf("## %s\n\n%s\n\n", doc.Header, doc.Content),
					//Prompt: fmt.Sprintf("## %s\n\n%s\n\n", doc.Header, doc.Content),
				},
				documentPath+"-"+strconv.Itoa(idx),
			)
			if err != nil {
				fmt.Println("😡:", err)
			} else {
				//embedding.MetaData = "📝 chunk num: " + strconv.Itoa(idx)
				_, err := store.Save(daphnia.VectorRecord{
					Prompt: embedding.Prompt,
					Embedding: embedding.Embedding,
					Id: fmt.Sprintf("%d", idx),
				})
				if err != nil {
					fmt.Println("😡:", err)
				}
			}

			ui.Println(colors.Yellow, "---[Improved chunk]--------------------------------")
			ui.Println(colors.Yellow, embedding.Prompt)
			ui.Println(colors.Yellow, "---------------------------------------------------")

		}

		return nil
	})

}

func chatWithCharacter(storePath, ollamaUrl, embeddingsModel, chatModel, instructionsPath string) {
	store := daphnia.VectorStore{}
	err := store.Initialize(storePath)

	//store := embeddings.BboltVectorStore{}
	//err := store.Initialize(storePath)
	if err != nil {
		log.Fatalln("😡:", err)
	}

	// Read the content of the file
	instructions, err := content.ReadTextFile(instructionsPath)
	if err != nil {
		log.Fatalln("😡:", err)
	}

	options := llm.SetOptions(map[string]interface{}{
		option.Temperature:   0.0,
		option.RepeatLastN:   2,
		option.RepeatPenalty: 3.0,
		option.TopK:          10,
		option.TopP:          0.5,
	})

	memory := []llm.Message{}

	for {

		question, _ := ui.Input(colors.Cyan, fmt.Sprintf("🤖 [%s] ask me something> ", chatModel))

		if question == "bye" {
			break
		}

		// Create an embedding from the question
		embeddingFromQuestion, err := embeddings.CreateEmbedding(
			ollamaUrl,
			llm.Query4Embedding{
				Model:  embeddingsModel,
				Prompt: question,
			},
			"question",
		)
		if err != nil {
			log.Fatalln("😡:", err)
		}
		fmt.Println("🔎 searching for similarity...")

		vector := daphnia.VectorRecord{
			Prompt: embeddingFromQuestion.Prompt,
			Embedding: embeddingFromQuestion.Embedding,
		}

		//similarities, _ := store.SearchSimilarities(embeddingFromQuestion, 0.3)
		similarities, _ := store.SearchTopNSimilarities(vector, 0.3, 2)

		fmt.Println("🎉 number of similarities:", len(similarities))

		//documentsContent := embeddings.GenerateContextFromSimilarities(similarities)
		documentsContent := "<context>\n"
		for _, similarity := range similarities {
			documentsContent += fmt.Sprintf("<doc>%s</doc>\n", similarity.Prompt)
		}
		documentsContent += "</context>"

		fmt.Println(documentsContent)

		messages := []llm.Message{
			{Role: "system", Content: instructions},
			{Role: "system", Content: documentsContent},
			//{Role: "user", Content: question},
		}

		messages = append(messages, memory...)
		messages = append(messages, llm.Message{Role: "user", Content: question})

		queryChat := llm.Query{
			Model:    chatModel,
			Messages: messages,
			Options:  options,
		}

		fmt.Println()
		ui.Println(colors.Magenta, "🤖 answer:")

		// Answer the question
		lastAnswer, err := completion.ChatStream(ollamaUrl, queryChat,
			func(answer llm.Answer) error {
				fmt.Print(answer.Message.Content)
				return nil
			})

		if err != nil {
			log.Fatal("😡:", err)
		}

		// Save the conversation in memory
		memory = append(
			memory,
			llm.Message{Role: "user", Content: question},
			llm.Message{Role: "assistant", Content: lastAnswer.Message.Content},
		)

		// Then, if the length is greater than 2, remove the two oldest messages
		if len(memory) > 2 {
			memory = memory[2:]
		}

		fmt.Println()

		//ui.Println(colors.Orange, answer.Message.Content)
	}

}

/*

go run main.go create-embeddings \
--embeddings-model bge-m3:latest \
--contextual-model qwen2.5:3b \
--docs-path ./docs \
--store-path ./embeddings.db \
--url http://host.docker.internal:11434


go run main.go chat \
--embeddings-model bge-m3:latest \
--chat-model qwen2.5:0.5b \
--store-path ./embeddings.db \
--context-path ./context.md \
--instructions-path ./instructions.md \
--url http://host.docker.internal:11434

----------------------------------------------------
go run main.go create-embeddings \
--embeddings-model bge-m3:latest \
--contextual-model granite3-moe:3b \
--docs-path ./docs \
--store-path ./embeddings_2.db \
--url http://host.docker.internal:11434


go run main.go chat \
--embeddings-model bge-m3:latest \
--chat-model granite3-moe:1b \
--store-path ./embeddings_2.db \
--instructions-path ./instructions.md \
--url http://host.docker.internal:11434

----------------------------------------------------
granite3-moe:3b                                          5d8ebcfcdb80    2.1 GB    2 days ago      
granite3-moe:1b                                          881c120886f6    821 MB    2 days ago      
bge-m3:latest                                            790764642607    1.2 GB    2 days ago 

go run main.go create-embeddings \
--embeddings-model bge-m3:latest \
--contextual-model granite3-moe:3b \
--docs-path ./docs \
--store-path ./embeddings_parakeet.db \
--url http://host.docker.internal:11434

go run main.go chat \
--embeddings-model bge-m3:latest \
--chat-model granite3-moe:3b \
--store-path ./embeddings_parakeet.db \
--store-path ./embeddings_parakeet.db \
--url http://host.docker.internal:11434

----------------------------------------------------



--- with a RPi ---

go run main.go create-embeddings \
--embeddings-model mxbai-embed-large:latest  \
--contextual-model qwen2.5:1.5b \
--docs-path ./docs \
--store-path ./embeddings.db \
--url http://bob.local:11434

--url http://localhost:11434


go run main.go chat \
--embeddings-model mxbai-embed-large:latest \
--chat-model qwen2.5:0.5b \
--store-path ./embeddings.db \
--instructions-path ./instructions.md \
--url http://bob.local:11434

--url http://localhost:11434

*/

/*
- add flags for a config file (or string) for options
- number of top similarities
- verbose mode
- set the number of messages to keep in memory

*/

func main() {

	// default values
	ollamaUrl := "http://localhost:11434"
	embeddingsModel := "bge-m3:latest"
	storePath := "./embeddings.db"
	docsPath := "./docs"
	contextualModel := "tinydolphin"

	chatModel := "llama3.1:8b"
	instructionsPath := "./instructions.md"

	args, flags := cli.Settings()

	if cli.HasFlag("url", flags) {
		ollamaUrl = cli.FlagValue("url", flags)
	}
	if cli.HasFlag("embeddings-model", flags) {
		embeddingsModel = cli.FlagValue("embeddings-model", flags)
	}
	if cli.HasFlag("store-path", flags) {
		storePath = cli.FlagValue("store-path", flags)
	}
	if cli.HasFlag("docs-path", flags) {
		docsPath = cli.FlagValue("docs-path", flags)
	}
	if cli.HasFlag("contextual-model", flags) {
		contextualModel = cli.FlagValue("contextual-model", flags)
	}

	if cli.HasFlag("chat-model", flags) {
		chatModel = cli.FlagValue("chat-model", flags)
	}

	if cli.HasFlag("instructions-path", flags) {
		instructionsPath = cli.FlagValue("instructions-path", flags)
	}

	switch cmd := cli.ArgsTail(args); cmd[0] {
	case "create-embeddings":
		createEmbeddings(docsPath, storePath, ollamaUrl, embeddingsModel, contextualModel)
	case "chat":
		chatWithCharacter(storePath, ollamaUrl, embeddingsModel, chatModel, instructionsPath)
	default:
		fmt.Println("Unknown command:", cmd[0])
	}

}
