import express from "express"
import axios from "axios"
import cors from "cors"
import "dotenv/config"

const app = express()
const PORT = process.env.PORT || 8000

app.use(cors())

console.log(process.env.EDAMAM_KEY)

// app.get('/recipes/chicken', async (req, res) => {
//     const response = await axios.get(
//         `https://api.edamam.com/api/recipes/v2?type=public&q=chicken&app_id=${process.env.EDAMAM_ID}&app_key=${process.env.EDAMAM_KEY}`
//     )
//     console.log(response.data.hits)
//     res.json(response.data.hits)
// })

app.get('/recipes/:query', async (req, res) => {
    const response = await axios.get(
        `https://api.edamam.com/api/recipes/v2?type=public&q=${req.params.query}&app_id=1b60eec7&app_key=e4ec8ffc5d3894c38e4adff288f93e07`
    )
    console.log(response.data.hits)
    res.json(response.data.hits)
})

app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`)
})