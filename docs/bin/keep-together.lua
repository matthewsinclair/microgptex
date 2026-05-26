-- keep-together.lua
-- Pandoc Lua filter that handles .keep-together divs
-- Wraps content in Typst's unbreakable block

function Div(el)
  if el.classes:includes("keep-together") then
    local blocks = el.content
    -- Wrap in Typst unbreakable block
    table.insert(blocks, 1, pandoc.RawBlock("typst", "#block(breakable: false)["))
    table.insert(blocks, pandoc.RawBlock("typst", "]"))
    return blocks
  end
end
