-- fix-table-widths.lua
-- Pandoc Lua filter that removes fixed column widths from tables
-- This allows Typst to auto-size columns based on content

function Table(tbl)
  if tbl.colspecs then
    for i, colspec in ipairs(tbl.colspecs) do
      -- Keep alignment (colspec[1]) but remove width (colspec[2])
      tbl.colspecs[i] = {colspec[1], nil}
    end
  end
  return tbl
end
